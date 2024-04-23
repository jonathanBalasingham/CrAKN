from typing import List

import dgl
import amd
import torch
from torch import nn
from torch.utils.data import SubsetRandomSampler, DataLoader, SequentialSampler, Sampler
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
from crakn.backbones import *
from crakn.config import TrainingConfig
from scipy.spatial import KDTree
import warnings


class IndexSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)


def find_nearest_neighbors(comp, struct,
                           test_comp, test_struct,
                           k: float = None,
                           cutoff: float = None,
                           strategy: str = "k-nearest") -> List[np.array]:
    nearest_neighbors = []
    assert comp.shape[0] == struct.shape[0]
    comp_tree = KDTree(comp)
    struct_tree = KDTree(struct)
    if strategy == "k-nearest":
        all_comp = np.vstack([comp, test_comp])
        all_struct = np.vstack([struct, test_struct])
        comp_neighbors = comp_tree.query(all_comp, k=k)[1]
        struct_neighbors = struct_tree.query(all_struct, k=k)[1]
        neighbors = np.hstack([comp_neighbors, struct_neighbors])
        nearest_neighbors = [np.unique(i) for i in neighbors]
    elif strategy == "cutoff":
        comp_neighbors = comp_tree.query_ball_point(np.vstack([comp, test_comp]), r=cutoff)
        struct_neighbors = struct_tree.query_ball_point(np.vstack([struct, test_struct]), r=cutoff)
        nearest_neighbors = [np.unique(np.hstack([i, j]))
                             for i, j in zip(comp_neighbors, struct_neighbors)]
    else:
        raise NotImplementedError(f"Strategy: {strategy} not valid")
    return nearest_neighbors


def nearest_neighbors_to_edge_list(neighbors):
    edge_list = []
    for i, nn_list in enumerate(neighbors):
        for n in nn_list:
            edge_list.append([i, n])
    return edge_list


def knowledge_graph(train_dataset, val_dataset, test_dataset, config: TrainingConfig) -> dgl.graph:
    num_nodes = len(train_dataset) + len(val_dataset) + len(test_dataset)
    g = dgl.graph(data=[])
    g.add_nodes(num_nodes)
    comp_struct = np.vstack([train_dataset.amds, val_dataset.amds])
    comp = comp_struct[:, :200]
    struct = comp_struct[:, 200:]

    test_comp_struct = np.vstack([test_dataset.amds])
    test_comp = test_comp_struct[:, :200]
    test_struct = test_comp_struct[:, 200:]

    edge_list = nearest_neighbors_to_edge_list(
        find_nearest_neighbors(comp, struct,
                               test_comp, test_struct,
                               strategy=config.neighbor_strategy,
                               k=config.max_neighbors,
                               cutoff=config.cutoff)
    )

    edge_list = torch.Tensor(edge_list)
    u = torch.concat([edge_list[:, 0], edge_list[:, 1]], dim=0).type(torch.int64)
    v = torch.concat([edge_list[:, 1], edge_list[:, 0]], dim=0).type(torch.int64)

    g.add_edges(u=u, v=v)

    struct_feat = torch.Tensor(np.vstack([struct, test_struct]))
    comp_feat = torch.Tensor(np.vstack([comp, test_comp]))

    g.edata["comp"] = comp_feat[u] - comp_feat[v]
    g.edata["struct"] = struct_feat[u] - struct_feat[v]

    g.ndata["node_features"] = torch.Tensor(
        np.vstack([
            train_dataset.data,
            val_dataset.data,
            test_dataset.data
        ])
    )
    g.ndata["lattice"] = torch.Tensor(
        np.vstack([
            train_dataset.lattices,
            val_dataset.lattices,
            test_dataset.lattices
        ])
    )

    targets = torch.Tensor(
        np.concatenate([
            train_dataset.targets,
            val_dataset.targets,
            test_dataset.targets
        ], axis=0)
    )

    ids = train_dataset.ids + val_dataset.ids + test_dataset.ids

    indices = [i for i in range(targets.shape[0])]
    train_indices = indices[:len(train_dataset)]
    val_indices = indices[-(len(val_dataset) + len(test_dataset)):-len(test_dataset)]
    test_indices = indices[-len(test_dataset):]
    dataset = CrAKNGraphDataset(g, targets, ids)
    collate_fn = dataset.collate_fn

    train_loader = DataLoader(dataset, batch_size=config.batch_size,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=config.num_workers,
                              collate_fn=collate_fn, pin_memory=config.pin_memory,
                              shuffle=False)
    val_loader = DataLoader(dataset, batch_size=config.batch_size,
                            sampler=SubsetRandomSampler(val_indices),
                            num_workers=config.num_workers,
                            collate_fn=collate_fn, pin_memory=config.pin_memory,
                            shuffle=False)

    test_loader = DataLoader(dataset, batch_size=config.test_batch_size,
                             sampler=IndexSampler(test_indices),
                             num_workers=config.num_workers,
                             collate_fn=collate_fn, pin_memory=config.pin_memory)
    return train_loader, val_loader, test_loader


class CrAKNGraphDataset(torch.utils.data.Dataset):

    def __init__(self, g: dgl.graph, targets, ids):
        super().__init__()
        self.graph = g
        self.targets = targets
        self.ids = ids

    def __len__(self):
        """Get length."""
        return self.graph.num_nodes()

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        sg, original_ids = self.graph.khop_out_subgraph(idx, k=1, relabel_nodes=True)
        return (sg,
                original_ids,
                torch.IntTensor([idx]),
                self.ids[idx],
                torch.Tensor([self.targets[idx]]))

    def collate_fn(self, samples):
        graphs, original_ids, target_ids, cids, targets = map(list, zip(*samples))
        #batched_graph = dgl.batch(graphs)
        batched_graph, original_ids = self.graph.khop_out_subgraph(target_ids, k=1, relabel_nodes=True)
        return (batched_graph,
                original_ids,
                torch.IntTensor(target_ids),
                cids,
                torch.vstack(targets))

