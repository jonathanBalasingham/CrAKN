"""A baseline graph convolution network dgl implementation."""
import dgl
import torch
from dgl.nn import AvgPooling, GraphConv, SumPooling
from typing import Literal
from torch import nn
from torch.nn import functional as F

from ..utils import BaseSettings
from pydantic_settings import SettingsConfigDict
from pymatgen.core.structure import Structure
from .graphs import ddg
from typing import Tuple, List
from .pst import AtomFeaturizer, DistanceExpansion


class SimpleGCNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.gcn."""

    name: Literal["SimpleGCN"]
    atom_encoding: Literal["mat2vec", "cgcnn"] = "mat2vec"
    atom_input_features: int = 200
    max_neighbors: int = 12
    weight_edges: bool = True
    layers: int = 1
    embedding_features: int = 64
    output_features: int = embedding_features
    edge_lengthscale: int = 8
    use_cuda: bool = torch.cuda.is_available()
    backward_edges: bool = True
    model_config = SettingsConfigDict(env_prefix="jv_model")


class SimpleGCN(nn.Module):
    """GraphConv GCN with DenseNet-style connections."""

    def __init__(
            self, config: SimpleGCNConfig = SimpleGCNConfig(name="SimpleGCN")
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.edge_lengthscale = config.edge_lengthscale
        self.weight_edges = config.weight_edges

        if config.atom_encoding not in ["mat2vec", "cgcnn"]:
            raise ValueError(f"atom_encoding_dim must be in {['mat2vec', 'cgcnn']}")
        else:
            atom_encoding_dim = 200 if config.atom_encoding == "mat2vec" else 92
            id_prop_file = "mat2vec.csv" if config.atom_encoding == "mat2vec" else "atom_init.json"

        self.af = AtomFeaturizer(use_cuda=config.use_cuda, id_prop_file=id_prop_file)

        self.atom_embedding = nn.Linear(
            atom_encoding_dim, config.embedding_features
        )

        self.layer1 = [GraphConv(config.embedding_features, config.embedding_features) for _ in range(config.layers)]

        self.layer2 = GraphConv(config.embedding_features, config.output_features)
        self.readout = AvgPooling()
        self.weighted_readout = SumPooling()

    def forward(self, g: dgl.graph):
        """Baseline SimpleGCN : start with `atom_features`."""
        if isinstance(g, (tuple, list)):
            g = g[0]
        g = g.local_var()
        print(g.num_nodes())

        if self.weight_edges:
            r = g.edata["distances"]
            edge_weights = torch.exp(-(r ** 2) / self.edge_lengthscale ** 2)
        else:
            edge_weights = None

        # initial node features: atom feature network...
        v = g.ndata.pop("atom_types")
        node_features = self.atom_embedding(self.af(v))

        x = node_features
        for layer in self.layer1:
            x = F.relu(layer(g, node_features, edge_weight=edge_weights))
        x = self.layer2(g, x, edge_weight=edge_weights)

        print(x.shape)
        if "weights" in g.ndata:
            out = self.weighted_readout(g, g.ndata["weights"] * x)
        else:
            out = self.readout(g, x)
        print(out.shape)
        return out


class GCNData(torch.utils.data.Dataset):

    def __init__(self, structures, targets, config: SimpleGCNConfig):
        self.atom_input_features = config.atom_input_features
        self.edge_lengthscale = config.edge_lengthscale
        self.id_prop_data = targets
        self.graphs = [ddg(structure, max_neighbors=config.max_neighbors, backward_edges=config.backward_edges) for
                       structure in structures]

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx], self.id_prop_data[idx]
        return self.graphs[idx], torch.Tensor([float(target)])

    @staticmethod
    def collate_fn(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.stack(labels, dim=0)

    @staticmethod
    def prepare_batch(batch: Tuple[dgl.DGLGraph, torch.Tensor], device=None, non_blocking=False, subset=None):
        """Send batched dgl crystal graph to device."""

        g, t = batch
        if subset is not None:
            return (
                g.subgraph(g.nodes()[:subset]).to(device=device, non_blocking=non_blocking),
                t[:subset].to(device=device, non_blocking=non_blocking)
            )

        return (
            g.to(device, non_blocking=non_blocking),
            t.to(device, non_blocking=non_blocking),
        )
