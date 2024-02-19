"""A baseline graph convolution network dgl implementation."""
# import dgl
import torch
from dgl.nn import AvgPooling, GraphConv
from typing import Literal
from torch import nn
from torch.nn import functional as F

from ..utils import BaseSettings
from pydantic_settings import SettingsConfigDict


class SimpleGCNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.gcn."""

    name: Literal["simplegcn"]
    atom_input_features: int = 1
    weight_edges: bool = True
    layers: int = 1
    embedding_features: int = 64
    output_features: int = 1
    edge_lengthscale: int = 8
    model_config = SettingsConfigDict(env_prefix="jv_model")


class SimpleGCN(nn.Module):
    """GraphConv GCN with DenseNet-style connections."""

    def __init__(
        self, config: SimpleGCNConfig = SimpleGCNConfig(name="simplegcn")
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.edge_lengthscale = config.edge_lengthscale
        self.weight_edges = config.weight_edges

        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.embedding_features
        )

        self.layer1 = [GraphConv(config.embedding_features, config.embedding_features) for _ in range(config.layers)]

        self.layer2 = GraphConv(config.embedding_features, config.output_features)
        self.readout = AvgPooling()

    def forward(self, g):
        """Baseline SimpleGCN : start with `atom_features`."""
        if isinstance(g, (tuple, list)):
            g = g[0]
        g = g.local_var()

        if self.weight_edges:
            r = torch.norm(g.edata["r"], dim=1)
            edge_weights = torch.exp(-(r ** 2) / self.edge_lengthscale ** 2)
        else:
            edge_weights = None

        # initial node features: atom feature network...
        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(v)

        x = node_features
        for layer in self.layer1:
            x = F.relu(layer(g, node_features, edge_weight=edge_weights))
        x = self.layer2(g, x, edge_weight=edge_weights)
        x = self.readout(g, x)

        return torch.squeeze(x)
