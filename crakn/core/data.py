import random
from functools import partial, reduce
from math import nan
from pathlib import Path

import amd
import jarvis.core.atoms
import numpy as np
from scipy.spatial import KDTree
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from crakn.backbones.cgcnn import CGCNNData
from crakn.backbones.gcn import GCNData
from crakn.backbones.matformer import MatformerData
from crakn.backbones.pst import PSTData
from crakn.backbones.pst_v2 import PSTv2Data
from crakn.config import TrainingConfig

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from typing import List, Tuple
from pymatgen.core.structure import Structure
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import pandas as pd
# from matbench.bench import MatbenchBenchmark
import torch

from crakn.core.graph import knowledge_graph

DATA_FORMATS = {
    "PST": "pymatgen",
    "Matformer": "jarvis",
    "CGCNN": "jarvis",
    "SimpleGCN": "pymatgen",
    "PSTv2": "pymatgen"
}


def retrieve_data(config: TrainingConfig) -> Tuple[List[Structure], List[List[float]], List]:
    if config.dataset == "matbench":
        raise NotImplementedError("Matbench not implemented")
    else:
        d = data(config.dataset)

    structures: List[Structure] = []
    targets: List[List[float]] = []
    ids = []
    current_id = 0
    for datum in tqdm(d, desc="Retrieving data.."):
        for t in config.target:
            if t not in datum.keys():
                raise ValueError(f"Unknown target {config.target}")

        target = []
        for property_name in config.target:
            val = datum[property_name]
            if isinstance(val, float):
                target.append(val)
            else:
                target.append(nan)

        if np.all(np.isnan(target)):
            continue

        if config.id_tag in datum:
            ids.append(datum[config.id_tag])
        else:
            ids.append(current_id)
            current_id += 1
        atoms = (Atoms.from_dict(datum["atoms"]) if isinstance(datum["atoms"], dict) else datum["atoms"])
        if DATA_FORMATS[config.base_config.backbone] == "pymatgen":
            structure = atoms.pymatgen_converter()
            structures.append(structure)
        elif DATA_FORMATS[config.base_config.backbone] == "jarvis":
            structures.append(atoms)
        targets.append(target)
    return structures, targets, ids


def get_dataset(structures, targets, ids, config):
    if config.name == "PST":
        return PSTData(structures, targets, ids, config)
    elif config.name == "PSTv2":
        return PSTv2Data(structures, targets, ids, config)
    elif config.name == "SimpleGCN":
        return GCNData(structures, targets, ids, config)
    elif config.name == "Matformer":
        return MatformerData(structures, targets, ids, config)
    elif config.name == "CGCNN":
        return CGCNNData(structures, targets, ids, config)
    else:
        raise NotImplementedError(f"Not implemented yet, {config.name}")


class CrAKNDataset(torch.utils.data.Dataset):

    def __init__(self, structures, targets, ids, config: TrainingConfig):
        super().__init__()
        self.data = get_dataset(structures, targets, ids, config.base_config.backbone_config)

        if isinstance(structures[0], jarvis.core.atoms.Atoms):
            structures = [s.pymatgen_converter() for s in structures]
        periodic_sets = [amd.periodicset_from_pymatgen_structure(s) for s in
                         tqdm(structures, desc="Creating Periodic Sets..")]
        if config.composition_features == "mat2vec":
            af = pd.read_csv(Path(__file__).parent.parent / "data" / "mat2vec.csv").to_numpy()[:, 1:].astype(
                np.float64)
        else:
            import json
            af = json.load(open(Path(__file__).parent.parent / "data" / "atom_init.json"))
            af = np.vstack(list(af.values()))

        comp = np.vstack([np.mean(af[ps.types - 1], axis=0) for ps in periodic_sets])
        amds = np.vstack([amd.AMD(ps, k=config.base_config.amd_k)
                          for ps in tqdm(periodic_sets, desc="Calculating AMDs..")])
        self.amds = np.concatenate([comp, amds], axis=1)
        self.lattices = np.array([list(s.lattice.parameters)
                                  for s in tqdm(structures, desc="Retrieving lattices..")])
        self.targets = targets
        self.ids = ids

    def __len__(self):
        """Get length."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        return (self.data[idx], torch.Tensor(self.amds[idx]), torch.Tensor(self.lattices[idx]),
                self.ids[idx], torch.Tensor(self.targets[idx]))


def convert(vlm: torch.nn.Module, loader: torch.utils.data.DataLoader, target_index):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    node_features = []
    AMDs = []
    lattices = []
    targets = []
    cids = []
    preds = []
    with torch.no_grad():
        for datum in loader:
            bb_data, amds, latt, ids, target = datum
            inds = np.where(np.logical_not(np.isnan(target[:, target_index])))[0]
            if inds.shape == target.shape[0]:
                continue

            amds = amds[inds]
            latt = latt[inds]
            ids = [ids[i] for i in inds]
            target = target[inds]
            train_inputs = bb_data[0]

            if isinstance(train_inputs, list) or isinstance(train_inputs, tuple):
                train_inputs = [i.to(device) for i in train_inputs]
            else:
                train_inputs = train_inputs.to(device)

            with torch.no_grad():
                nf = vlm(train_inputs, output_level="atom")
                base_preds = vlm(train_inputs, output_level="property")
                if isinstance(nf, tuple):
                    nf = [i.cpu() for i in nf]
                else:
                    nf = nf.cpu()
                base_preds = base_preds.cpu()

            node_features.append(nf)
            preds.append(base_preds)
            lattices.append(latt)
            AMDs.append(amds)
            targets.append(target[:, target_index])
            cids.append(ids)

    base_preds = torch.concat(preds, dim=0)
    AMDs = torch.concat(AMDs, dim=0)
    lattices = torch.concat(lattices, dim=0)
    targets = torch.concat(targets, dim=0)
    cids = reduce(lambda x, y: x + y, cids)

    return PretrainCrAKNDataset(node_features, AMDs, lattices, targets, cids, base_preds)


def convert_to_pretrain_dataset(vlm: torch.nn.Module,
                                train_loader: torch.utils.data.DataLoader,
                                val_loader: torch.utils.data.DataLoader,
                                test_loader: torch.utils.data.DataLoader,
                                config: TrainingConfig):
    d = data(config.dataset)
    train_dataset = convert(vlm, train_loader, target_index=config.mo_target_index)
    val_dataset = convert(vlm, val_loader, target_index=config.mo_target_index)
    test_dataset = convert(vlm, test_loader, target_index=config.mo_target_index)

    if config.base_config.mtype == "GNN":
        return knowledge_graph(train_dataset, val_dataset, test_dataset, config, d)

    return (DataLoader(dataset, batch_size=config.batch_size,
                       sampler=SubsetRandomSampler([i for i in range(len(dataset))]),
                       num_workers=config.num_workers,
                       collate_fn=collate_pretrain_crakn_data, pin_memory=config.pin_memory,
                       shuffle=False) for dataset in [train_dataset, val_dataset, test_dataset])


class PretrainCrAKNDataset(torch.utils.data.Dataset):

    def __init__(self, vlm_output, amds, lattices, targets, ids, base_preds):
        super().__init__()
        self.data = vlm_output
        self.amds = amds
        self.lattices = lattices
        self.targets = targets
        self.ids = ids
        self.base_preds = base_preds

    def __len__(self):
        """Get length."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        return (self.data[idx], torch.Tensor(self.amds[idx]), torch.Tensor(self.lattices[idx]),
                self.ids[idx], torch.Tensor([self.targets[idx]]))


def collate_crakn_data(dataset_list, internal_collate=PSTData.collate_fn):
    backbone_data = []
    lattices = []
    amds = []
    targets = []
    ids = []

    for bd, amd_fea, latt, datum_id, target in dataset_list:
        backbone_data.append(bd)
        amds.append(amd_fea)
        lattices.append(latt)
        targets.append(target)
        ids.append(datum_id)

    return (internal_collate(backbone_data),
            torch.stack(amds, dim=0),
            torch.stack(lattices, dim=0),
            ids,
            torch.stack(targets, dim=0))


def collate_pretrain_crakn_data(dataset_list):
    node_features = []
    lattices = []
    amds = []
    targets = []
    ids = []

    for nf, amd_fea, latt, datum_id, target in dataset_list:
        node_features.append(nf)
        amds.append(amd_fea)
        lattices.append(latt)
        targets.append(target)
        ids.append(datum_id)

    return (pad_sequence(node_features, batch_first=True),
            torch.stack(amds, dim=0),
            torch.stack(lattices, dim=0),
            ids,
            torch.stack(targets, dim=0))


def prepare_crakn_batch(batch, device=None, internal_prepare_batch=None, non_blocking=False, variable=True):
    if variable:
        subset = random.randrange(2, len(batch[-1]))
    else:
        subset = len(batch[-1])

    batch = (
        (internal_prepare_batch(batch[0], device=device, non_blocking=non_blocking, subset=subset),
         batch[1][:subset].to(device=device, non_blocking=non_blocking),
         batch[2][:subset].to(device=device, non_blocking=non_blocking),
         batch[3][:subset]),
        batch[4][:subset].to(device=device, non_blocking=non_blocking),
    )
    return batch


def get_dataloader(dataset: CrAKNDataset, config: TrainingConfig):
    total_size = len(dataset)
    random.seed(config.random_seed)
    if config.n_train is None:
        if config.train_ratio is None:
            assert config.val_ratio + config.test_ratio < 1
            train_ratio = 1 - config.val_ratio - config.test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert config.train_ratio + config.val_ratio + config.test_ratio <= 1
            train_ratio = config.train_ratio
    indices = list(range(total_size))
    if not config.keep_data_order:
        random.shuffle(indices)

    if config.n_train:
        train_size = config.n_train
    else:
        train_size = int(train_ratio * total_size)
    if config.n_test:
        test_size = config.n_test
    else:
        test_size = int(config.test_ratio * total_size)
    if config.n_val:
        valid_size = config.n_val
    else:
        valid_size = int(config.val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])

    collate_fn = partial(collate_crakn_data, internal_collate=dataset.data.collate_fn)

    train_loader = DataLoader(dataset, batch_size=config.batch_size,
                              sampler=train_sampler,
                              num_workers=config.num_workers,
                              collate_fn=collate_fn, pin_memory=config.pin_memory,
                              shuffle=False)

    val_loader = DataLoader(dataset, batch_size=config.batch_size,
                            sampler=val_sampler,
                            num_workers=config.num_workers,
                            collate_fn=collate_fn, pin_memory=config.pin_memory,
                            shuffle=False)

    """test_loader = DataLoader(dataset, batch_size=config.batch_size,
                             sampler=test_sampler,
                             num_workers=config.num_workers,
                             collate_fn=collate_fn, pin_memory=config.pin_memory)"""
    test_set = torch.utils.data.Subset(dataset, indices[-test_size:])
    test_loader = DataLoader(test_set, batch_size=config.test_batch_size,
                             # sampler=test_sampler,
                             num_workers=config.num_workers,
                             collate_fn=collate_fn, pin_memory=config.pin_memory)
    return train_loader, val_loader, test_loader


def create_test_dataloader(net: nn.Module, train_loader, test_loader, prepare_batch, max_neighbors):
    neighbor_data = []
    for dat in tqdm(train_loader, desc="Generating knowledge network node features.."):
        X, target = prepare_batch(dat)
        neighbor_node_features = net(X, return_embeddings=True)
        neighbor_data.append((neighbor_node_features, X[1], X[2], target))

    test_data = []
    for dat in tqdm(test_loader, desc="Generating knowledge network node features.."):
        X, target = prepare_batch(dat)
        neighbor_node_features = net(X, return_embeddings=True)
        test_data.append((neighbor_node_features, X[1], X[2], target))

    tree = KDTree(torch.concat([i[0] for i in neighbor_data], dim=0).cpu())
    test_embeddings = torch.concat([i[0] for i in test_data]).cpu()
    nearest_neighbor_indices = tree.query(test_embeddings, k=max_neighbors)[1]
    td = []

    for i, test_datum in enumerate(test_loader.dataset):
        nni = nearest_neighbor_indices[i]
        nn_data = [train_loader.dataset[i] for i in nni]
        nn_data.append(test_datum)
        nn_data = list(zip(*nn_data))

        temp_data = (train_loader.dataset.data.collate_fn(nn_data[0]),
                     torch.stack(nn_data[1], dim=0),
                     torch.stack(nn_data[2], dim=0),
                     nn_data[3],
                     torch.stack(nn_data[4], dim=0))
        td.append(temp_data)

    return td


if __name__ == "__main__":
    fake_config = TrainingConfig()
    structures, targets, ids = retrieve_data(fake_config)
    dataset = CrAKNDataset(structures, targets, ids, fake_config)
