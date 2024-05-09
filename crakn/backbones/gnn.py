from pathlib import Path
from typing import Tuple, List

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.nn import AvgPooling
from dgl.nn.pytorch.conv import PNAConv
from typing import Literal

from dgl.nn.pytorch import SumPooling
from jarvis.core.specie import get_node_attributes, chem_data
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from crakn.backbones.graphs import Graph, ddg

# import torch
from crakn.backbones.utils import RBFExpansion, AtomFeaturizer, DistanceExpansion
from crakn.utils import BaseSettings
from pydantic_settings import SettingsConfigDict
from scipy.constants import e, epsilon_0


class GNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["GNN"]
    atom_encoding: Literal["mat2vec", "cgcnn"] = "mat2vec"
    conv_layers: int = 3
    atom_input_features: int = 92
    edge_features: int = 40
    embedding_features: int = 256
    output_features: int = embedding_features
    outputs: int = 1
    neighbor_strategy: str = "k-nearest"
    cutoff: float = 8.0
    max_neighbors: int = 12
    use_canonize: bool = False
    model_config = SettingsConfigDict(env_prefix="jv_model")


class GNNConv(nn.Module):

    def __init__(
            self,
            node_features: int = 64,
            edge_features: int = 32,
            return_messages: bool = False,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.return_messages = return_messages
        self.linear_q = nn.Sequential(
            nn.Linear(node_features, node_features)
        )

        self.linear_k = nn.Sequential(
            nn.Linear(node_features, node_features)
        )

        self.linear_v = nn.Sequential(
            nn.Linear(node_features, node_features)
        )

        self.linear_edge = nn.Sequential(
            nn.Linear(node_features, node_features)
        )

        self.linear_m = nn.Sequential(
            nn.Linear(node_features, node_features)
        )

        self.linear_g = nn.Sequential(
            nn.Linear(node_features, node_features),
            nn.Mish()
        )

        self.final = nn.Sequential(
            nn.Linear(node_features, node_features),
            nn.Mish(),
            nn.LayerNorm(node_features),
            nn.Linear(node_features, node_features)
        )

        self.ln = nn.LayerNorm(node_features)
        self.bn_message = nn.BatchNorm1d(node_features)
        self.register_buffer("e", torch.Tensor([e]).float())
        self.register_buffer("epsilon_0", torch.Tensor([epsilon_0]).float())

        self.bn = nn.BatchNorm1d(node_features)
        self.bn2 = nn.BatchNorm1d(node_features)

    def forward(
            self,
            g: dgl.DGLGraph,
            node_feats: torch.Tensor,
            edge_feats: torch.Tensor,
    ):
        g = g.local_var()

        g.ndata["q"] = self.linear_q(node_feats)
        g.ndata["k"] = self.linear_k(node_feats)
        g.ndata["v"] = self.linear_v(node_feats)

        e = self.linear_edge(edge_feats)
        m = self.linear_m(edge_feats)

        g.apply_edges(fn.u_sub_v("q", "k", "h_nodes"))
        g.edata["w"] = dgl.nn.functional.edge_softmax(g, m * g.edata["h_nodes"] + e)
        g.apply_edges(fn.u_mul_e("v", "w", "m"))

        g.update_all(
            message_func=fn.copy_e("m", "z"),
            reduce_func=fn.sum("z", "h"),
        )

        h = g.ndata.pop("h")
        h = self.final(h + node_feats)
        h = self.ln(h)
        return h, edge_feats


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
        self.abf = RBFExpansion(
            vmin=-np.pi / 2, vmax=np.pi / 2, bins=config.edge_features
        )

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

        self.edge_embedding = nn.Linear(
            config.edge_features, config.embedding_features
        )

        #self.conv_layers = nn.ModuleList(
        #    [
        #        GNNConv(config.embedding_features, config.edge_features)
        #        for _ in range(config.conv_layers)
        #    ]
        #)

        self.conv_layers = nn.ModuleList(
            [
                PNAConv(config.embedding_features,
                        config.embedding_features,
                        ['mean', 'var', 'min', 'max'],
                        ['identity', 'attenuation'],
                        np.log(config.max_neighbors + 1))
                for _ in range(config.conv_layers)
            ]
        )

        self.de = DistanceExpansion(
            size=config.edge_features,
        )

        self.readout = AvgPooling()
        self.weighted_readout = SumPooling()
        self.fc = nn.Sequential(
            nn.Linear(config.embedding_features, config.embedding_features), nn.Softplus()
        )

        self.fc_out = nn.Linear(
            config.embedding_features, config.outputs
        )

        self.register_buffer("e", torch.Tensor([e]).float())
        self.register_buffer("epsilon_0", torch.Tensor([epsilon_0]).float())
        self.register_buffer("bc", torch.Tensor([1.381e-23]).float())
        self.register_buffer("A", torch.Tensor([1.024e-23]).float())
        self.register_buffer("B", torch.Tensor([1.582e-26]).float())
        self.A, self.B = self.A / self.bc, self.B / self.bc

    def pot(self, r):
        return self.B / r**12 - self.A / r**6

    @staticmethod
    def pooling(distribution, x):
        pooled = torch.sum(x, dim=1) / torch.sum(distribution, dim=1)
        return pooled

    def forward(self, g, output_level: Literal["atom", "crystal", "property"]) -> torch.Tensor:
        g = g.local_var()

        if "distance" not in g.edata:
            g.edata["distance"] = torch.norm(g.edata["r"], dim=-1, keepdim=False)

        edge_features = self.de(torch.reciprocal(g.edata["distance"]))
        edge_features = self.edge_embedding(edge_features)

        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(self.af(v))
        if "distances" in g.ndata:
            node_features = self.norm(node_features + self.distance_embedding(g.ndata["distances"]))

        for conv_layer in self.conv_layers:
            node_features = conv_layer(g, node_features, edge_feat=edge_features)

        if output_level == "atom":
            g.ndata["node_features"] = node_features
            node_features = [i.ndata["node_features"] for i in dgl.unbatch(g)]
            node_features = pad_sequence(node_features, batch_first=True)
            weights = torch.sum(torch.abs(node_features), dim=-1, keepdim=True)
            weights[weights > 0] = 1
            return weights, node_features

        # crystal-level readout
        if "weights" in g.ndata:
            features = self.weighted_readout(g, node_features * g.ndata["weights"].reshape((-1, 1)))
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
            config: GNNConfig = GNNConfig(name="GNN")
    ):

        graphs = [ddg(s,
                      collapse_tol=1e-4,
                      max_neighbors=config.max_neighbors)
                  for s in tqdm(structures, desc="Creating graphs..")]
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
            return g.to(device=device, non_blocking=non_blocking), t[:subset].to(device=device, non_blocking=non_blocking)

        return g.to(device, non_blocking=non_blocking), t.to(device, non_blocking=non_blocking)
