import torch
from torch import nn
from .config import BaseSettings
from typing import Literal, Union
import dgl

from ..backbones.gcn import SimpleGCNConfig, SimpleGCN
from ..backbones.pst import PeriodicSetTransformer, PSTConfig


class CrAKNConfig(BaseSettings):
    name: Literal["crakn"]
    backbone: Literal["PST", "SimpleGCN"]
    backbone_config: Union[PSTConfig, SimpleGCNConfig] = PSTConfig()
    embedding_dim: int = 64
    layers: int = 2
    dropout: float = 0
    output_dim: int = 1


def get_backbone(bb: str, bb_config) -> nn.Module:
    if bb == "PST":
        return PeriodicSetTransformer(bb_config)
    elif bb == "SimpleGCN":
        return SimpleGCN(bb_config)
    else:
        raise NotImplementedError(f"Unknown backbone: {bb}")


def get_data_retriever(bb: str):
    if bb == "PST":
        return lambda g: (g.ndata["pdd"], g.ndata["composition"])
    else:
        raise NotImplementedError(f"Unknown backbone: {bb}")


class CrAKNLayer(nn.Module):

    def __init__(self, embedding_dim: int, activation=nn.Mish, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.activation = activation()
        self.dense_edge = nn.Linear(embedding_dim, embedding_dim)
        self.activation_edge = activation()
        self.conv = dgl.nn.GINEConv(embedding_dim)
        self.proj_out = nn.Linear(embedding_dim, embedding_dim)
        self.activation_out = activation()

    def forward(self, g: dgl.DGLGraph, node_features: torch.Tensor, edge_features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.dense(node_features))
        y = self.activation(self.dense_edge(edge_features))
        out = self.conv(g, x, y)
        return self.activation_out(self.proj_out(out))


class CrAKN(nn.Module):
    def __init__(self, config: CrAKNConfig):
        super().__init__()
        self.backbone = get_backbone(config.backbone, bb_config=config.backbone_config)
        self.retrieve_data = get_data_retriever(config.backbone)
        self.layers = [CrAKNLayer(config.embedding_dim) for _ in range(config.layers)]
        self.out = nn.Linear(config.embedding_dim, config.output_dim)
        self.bn = nn.BatchNorm1d(config.embedding_dim)
        self.pooling = dgl.nn.AvgPooling()

    def forward(self, g: dgl.graph) -> torch.Tensor:
        data = self.retrieve_data(g)
        node_features = self.backbone(data)
        edge_features = torch.pdist(g.ndata["amd"])
        target = g.ndata["target"]

        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, target)

        x = self.pooling(g, node_features)
        x = self.bn(x)
        return self.out(x)

