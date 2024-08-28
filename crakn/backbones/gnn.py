from pathlib import Path
from typing import Tuple, List

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
from dgl.nn import AvgPooling
from typing import Literal

from dgl.nn.pytorch import SumPooling
from jarvis.core.specie import get_node_attributes, chem_data
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from tqdm import tqdm

from crakn.backbones.WeightedBatchNorm1d import WeightedBatchNorm1d
from crakn.backbones.graphs import ddg, mdg

from crakn.backbones.utils import RBFExpansion, AtomFeaturizer, DistanceExpansion, GaussianExpansion
from crakn.utils import BaseSettings
from pydantic_settings import SettingsConfigDict


class GNNConfig(BaseSettings):
    name: Literal["GNN"]
    atom_encoding: Literal["mat2vec", "cgcnn"] = "mat2vec"
    conv_layers: int = 3
    atom_input_features: int = 92
    edge_features: int = 40
    embedding_features: int = 128
    output_features: int = embedding_features
    outputs: int = 1
    neighbor_strategy: Literal["ddg", "mdg"] = "mdg"
    cutoff: float = 8.0
    max_neighbors: int = 12
    dropout: float = 0.0
    backwards_edges: bool = True
    collapse_tol: float = 1e-4
    model_config = SettingsConfigDict(env_prefix="jv_model")


def aggregate_mean(h):
    """mean aggregation"""
    return torch.mean(h, dim=1)

def aggregate_max(h):
    """max aggregation"""
    return torch.max(h, dim=1)[0]

def aggregate_min(h):
    """min aggregation"""
    return torch.min(h, dim=1)[0]

def aggregate_sum(h):
    """sum aggregation"""
    return torch.sum(h, dim=1)

def aggregate_std(h):
    """standard deviation aggregation"""
    return torch.sqrt(aggregate_var(h) + 1e-30)

def aggregate_var(h):
    """variance aggregation"""
    h_mean_squares = torch.mean(h * h, dim=1)
    h_mean = torch.mean(h, dim=1)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var

def _aggregate_moment(h, n):
    """moment aggregation: for each node (E[(X-E[X])^n])^{1/n}"""
    h_mean = torch.mean(h, dim=1, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=1)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + 1e-30, 1. / n)
    return rooted_h_n

def aggregate_moment_3(h):
    """moment aggregation with n=3"""
    return _aggregate_moment(h, n=3)

def aggregate_moment_4(h):
    """moment aggregation with n=4"""
    return _aggregate_moment(h, n=4)

def aggregate_moment_5(h):
    """moment aggregation with n=5"""
    return _aggregate_moment(h, n=5)

def scale_identity(h):
    """identity scaling (no scaling operation)"""
    return h

def scale_amplification(h, D, delta):
    """amplification scaling"""
    return h * (np.log(D + 1) / delta)

def scale_attenuation(h, D, delta):
    """attenuation scaling"""
    return h * (delta / np.log(D + 1))


AGGREGATORS = {
    'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
    'std': aggregate_std, 'var': aggregate_var, 'moment3': aggregate_moment_3,
    'moment4': aggregate_moment_4, 'moment5': aggregate_moment_5
}
SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation
}


class PNAConvTower(nn.Module):

    def __init__(self, in_size, out_size, aggregators, scalers,
                 delta, dropout=0., edge_feat_size=0):
        super(PNAConvTower, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta
        self.edge_feat_size = edge_feat_size

        self.M = nn.Linear(2 * in_size + edge_feat_size, in_size)
        self.U = nn.Linear((len(aggregators) * len(scalers) + 1) * in_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(out_size)

    def reduce_func(self, nodes):
        msg = nodes.mailbox['msg']
        degree = msg.size(1)
        h = torch.cat([AGGREGATORS[agg](msg) for agg in self.aggregators], dim=1)
        h = torch.cat([
            SCALERS[scaler](h, D=degree, delta=self.delta) if scaler != 'identity' else h
            for scaler in self.scalers
        ], dim=1)
        return {'h_neigh': h}

    def message(self, edges):
        """message function for PNA layer"""
        if self.edge_feat_size > 0:
            f = torch.cat([edges.src['h'], edges.dst['h'], edges.data['a']], dim=-1)
        else:
            f = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)
        return {'msg': self.M(f)}

    def forward(self, graph, node_feat, edge_feat=None):
        snorm_n = torch.cat(
            [torch.ones(N, 1).to(node_feat) / N for N in graph.batch_num_nodes()],
            dim=0
        ).sqrt()
        with graph.local_scope():
            graph.ndata['h'] = node_feat
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata['a'] = edge_feat

            graph.update_all(self.message, self.reduce_func)
            h = self.U(
                torch.cat([node_feat, graph.ndata['h_neigh']], dim=-1)
            )
            h = h * snorm_n
            return self.dropout(self.batchnorm(h))


class PNAConv(nn.Module):

    def __init__(self, in_size, out_size, aggregators, scalers, delta,
                 dropout=0., num_towers=1, edge_feat_size=0, residual=True):
        super(PNAConv, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        assert in_size % num_towers == 0, 'in_size must be divisible by num_towers'
        assert out_size % num_towers == 0, 'out_size must be divisible by num_towers'
        self.tower_in_size = in_size // num_towers
        self.tower_out_size = out_size // num_towers
        self.edge_feat_size = edge_feat_size
        self.residual = residual
        if self.in_size != self.out_size:
            self.residual = False

        self.towers = nn.ModuleList([
            PNAConvTower(
                self.tower_in_size, self.tower_out_size,
                aggregators, scalers, delta,
                dropout=dropout, edge_feat_size=edge_feat_size
            ) for _ in range(num_towers)
        ])

        self.mixing_layer = nn.Sequential(
            nn.Linear(out_size, out_size),
            nn.LeakyReLU()
        )

    def forward(self, graph, node_feat, edge_feat=None):
        h_cat = torch.cat([
            tower(
                graph,
                node_feat[:, ti * self.tower_in_size: (ti + 1) * self.tower_in_size],
                edge_feat
            )
            for ti, tower in enumerate(self.towers)
        ], dim=1)
        h_out = self.mixing_layer(h_cat)
        # add residual connection
        if self.residual:
            h_out = h_out + node_feat

        return h_out


class GNN(nn.Module):

    def __init__(
            self, config: GNNConfig = GNNConfig(name="GNN")
    ):

        super().__init__()
        if config.atom_encoding not in ["mat2vec", "cgcnn"]:
            raise ValueError(f"atom_encoding_dim must be in {['mat2vec', 'cgcnn']}")
        else:
            atom_encoding_dim = 200 if config.atom_encoding == "mat2vec" else 92
            id_prop_file = "mat2vec.csv" if config.atom_encoding == "mat2vec" else "atom_init.json"

        self.rbf = RBFExpansion(vmin=0, vmax=1.0, bins=config.edge_features)

        self.af = AtomFeaturizer(
            use_cuda=torch.cuda.is_available(),
            id_prop_file=id_prop_file
        )

        self.atom_embedding = nn.Linear(
            atom_encoding_dim, config.embedding_features
        )

        self.distance_embedding = nn.Linear(
            config.max_neighbors, config.embedding_features
        )

        self.norm = nn.LayerNorm(config.embedding_features)
        self.wbn = WeightedBatchNorm1d(config.embedding_features)

        self.edge_embedding = nn.Linear(
            config.edge_features, config.embedding_features
        )

        self.conv_layers = nn.ModuleList(
            [
                PNAConv(config.embedding_features,
                        config.embedding_features,
                        ['mean', 'var', 'min', 'max'],
                        ['identity', 'attenuation', 'amplification'],
                        np.log(config.max_neighbors + 1),
                        num_towers=1,
                        dropout=config.dropout)
                for _ in range(config.conv_layers)
            ]
        )

        self.de = DistanceExpansion(
            size=config.edge_features,
        )

        self.ge = GaussianExpansion(size=config.edge_features)

        self.readout = AvgPooling()
        self.weighted_readout = SumPooling()
        self.fc = nn.Sequential(
            nn.Linear(config.embedding_features,
                      config.embedding_features),
            nn.Mish()
        )

        self.fc_out = nn.Linear(
            config.embedding_features, config.outputs
        )

    @staticmethod
    def pooling(distribution, x):
        pooled = torch.sum(x, dim=1) / torch.sum(distribution, dim=1)
        return pooled

    def forward(self, g, output_level: Literal["atom", "crystal", "property"]) -> torch.Tensor:
        g = g.local_var()
        g.ndata["weights"] = g.ndata["weights"][:, None]

        if "r" in g.edata:
            g.edata["distance"] = torch.norm(g.edata["r"], dim=-1, keepdim=False)

        if "moments" in g.edata:
            edge_features = self.ge(
                torch.reciprocal(g.edata["moments"][:, 0][:, None]),
                g.edata["moments"][:, 1][:, None]
            )
        else:
            edge_features = self.de(torch.reciprocal(g.edata["distance"]))

        edge_features = self.edge_embedding(edge_features)

        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(self.af(v))
        if "distances" in g.ndata:
            #encoding = torch.exp(-(g.ndata["distances"] ** 2))
            encoding = g.ndata["distances"]
            node_features = self.norm(node_features + self.distance_embedding(encoding))

        for conv_layer in self.conv_layers:
            node_features = conv_layer(g, node_features, edge_features)

        if output_level == "atom":
            g.ndata["node_features"] = node_features
            node_features = pad_sequence(
                [i.ndata["node_features"] for i in dgl.unbatch(g)],
                batch_first=True)
            weights = torch.sum(torch.abs(node_features), dim=-1, keepdim=True)
            weights[weights > 0] = 1
            return weights, node_features

        if "weights" in g.ndata:
            features = self.weighted_readout(g, node_features * g.ndata["weights"])
        else:
            features = self.readout(g, node_features)

        if output_level == "crystal":
            return features

        return self.fc_out(features)


class GNNData(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
            self,
            structures,
            targets,
            ids,
            graphs=None,
            config: GNNConfig = GNNConfig(name="GNN")
    ):
        if graphs is None:
            if config.neighbor_strategy == "mdg":
                graphs = [mdg(s,
                              max_neighbors=config.max_neighbors,
                              backward_edges=config.backwards_edges)
                          for s in tqdm(structures, desc=f"Creating {config.neighbor_strategy} graphs..")]
            else:
                graphs = [ddg(s,
                              max_neighbors=config.max_neighbors,
                              backward_edges=config.backwards_edges)
                          for s in tqdm(structures, desc=f"Creating {config.neighbor_strategy} graphs..")]

        self.graphs = graphs
        self.target = targets
        self.ids = ids

        self.labels = torch.tensor(targets).type(
            torch.get_default_dtype()
        )

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        if atom_features == "mat2vec":
            id_prop_file = "mat2vec.csv"
            path = Path(__file__).parent.parent / 'data' / id_prop_file
            return pd.read_csv(path).to_numpy()[:, 1:].astype("float32")
        else:
            template = get_node_attributes("C", atom_features)
            features = np.zeros((1 + max_z, len(template)))

            for element, v in chem_data.items():
                z = v["Z"]
                x = get_node_attributes(element, atom_features)

                if x is not None:
                    features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        return g, label

    @staticmethod
    def collate_fn(
            samples: List[Tuple[Data, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)

        if len(labels[0].size()) > 0:
            return batched_graph, torch.stack(labels)
        else:
            return batched_graph, torch.tensor(labels)

    @staticmethod
    def prepare_batch(batch: Tuple[dgl.graph, torch.Tensor], device=None, non_blocking=False, subset=None):
        g, t = batch
        if subset is not None:
            g = dgl.batch(dgl.unbatch(g)[:subset])
            return g.to(device=device, non_blocking=non_blocking), t[:subset].to(device=device,
                                                                                 non_blocking=non_blocking)

        return g.to(device, non_blocking=non_blocking), t.to(device, non_blocking=non_blocking)
