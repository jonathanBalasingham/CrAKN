import random

import torch
from torch import nn

from crakn.core.transformer_layers import CrAKNEncoder, CrAKNVectorAttention2D, CrAKNVectorAttention3D
from crakn.utils import BaseSettings
from typing import Literal, Union, List

from ..backbones.gcn import SimpleGCNConfig, SimpleGCN
from ..backbones.pst import PeriodicSetTransformer, PSTConfig
from ..backbones.matformer import MatformerConfig, Matformer
from ..backbones.cgcnn import CGCNN, CGCNNConfig
from ..backbones.pst_v2 import PeriodicSetTransformerV2, PSTv2Config


COMPONENTS = Literal[
    "vertex",
    "metavertex",
    "metaedge"
]

class CrAKNConfig(BaseSettings):
    name: Literal["crakn"]
    backbone: Literal["PST", "SimpleGCN", "Matformer", "PSTv2", "CGCNN"] = "PST"
    mtype: Literal["Transformer", "GNN"] = "GNN"
    backbone_config: Union[
        PSTConfig,
        SimpleGCNConfig,
        MatformerConfig,
        PSTv2Config,
        CGCNNConfig,
    ] = PSTConfig(name="PST")
    components: List[COMPONENTS] = ["vertex", "metavertex", "metaedge"]
    embedding_dim: int = 128
    layers: int = 1
    num_heads: int = 1
    head_dim: int = 128
    dropout: float = 0
    output_features: int = 1
    amd_k: int = 100
    comp_feat_size: int = 200
    extra_features: int = 4
    classification: bool = False
    backbone_only: bool = False
    attention_bias: bool = True
    embed_bias: bool = True
    cutoff: float = 1
    expansion_size: int = 128


def get_backbone(bb: str, bb_config) -> nn.Module:
    if bb == "PST":
        return PeriodicSetTransformer(bb_config)
    elif bb == "PSTv2":
        return PeriodicSetTransformerV2(bb_config)
    elif bb == "SimpleGCN":
        return SimpleGCN(bb_config)
    elif bb == "Matformer":
        return Matformer(bb_config)
    elif bb == "CGCNN":
        return CGCNN(bb_config)
    else:
        raise NotImplementedError(f"Unknown backbone: {bb}")


class CrAKNAdaptor(nn.Module):

    def __init__(self,
                 embedding_dim,
                 output_dim,
                 pooling,
                 layers=1,
                 attention_dropout=0.0,
                 qkv_bias=True):
        super().__init__()
        self.layers = nn.ModuleList([
            CrAKNVectorAttention3D(
                embedding_dim,
                attention_dropout=attention_dropout,
                qkv_bias=qkv_bias
            ) for _ in range(layers)])

        self.out = nn.Linear(embedding_dim, output_dim, bias=False)
        self.pooling = pooling

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.out(x)
        return x


class CrAKN(nn.Module):
    def __init__(self, config: CrAKNConfig):
        super().__init__()
        self.backbone = get_backbone(
            config.backbone,
            bb_config=config.backbone_config
        )

        self.vertex_embedding = CrAKNAdaptor(
            config.backbone_config.output_features,
            config.embedding_dim,
            self.backbone.pooling,
            layers=config.layers
        )

        self.components = config.components
        self.backbone_only = config.backbone_only
        self.attention_bias = config.attention_bias
        self.metric_comp_size = config.comp_feat_size
        self.metric_struct_size = config.amd_k

        self.metaedge_embedding = nn.Linear(
            self.metric_struct_size,
            config.embedding_dim
        )

        self.metavertex_embedding = nn.ModuleList([
            CrAKNVectorAttention2D(
                config.embedding_dim,
                use_bias=True,
                use_multiplier=False,
                attention_dropout=0.1)
            for _ in range(config.layers)])

        self.out = nn.Linear(
            config.backbone_config.output_features if config.backbone_only else config.embedding_dim,
            config.output_features
        )

        if self.backbone_only:
            self.backbone_out = nn.Linear(
                config.backbone_config.output_features,
                config.backbone_config.outputs
            )

        self.embedding = nn.Linear(
            config.backbone_config.output_features,
            config.embedding_dim
        )

        assert 0 <= config.dropout <= 1
        self.dropout = nn.Dropout(p=config.dropout)
        self.out = nn.Linear(
            config.embedding_dim,
            config.output_features
        )

    def forward(self, inputs, pretrained=False, return_embeddings=False) -> torch.Tensor:
        backbone_input, amds, latt, _ = inputs
        bb_X = backbone_input[:-1]
        if len(bb_X) == 1:
            bb_X = bb_X[0]

        if self.backbone_only:
            return self.backbone(bb_X, output_level="property")

        if pretrained:
            with torch.no_grad():
                distribution, vertex_features = self.backbone(bb_X, output_level="atom")
        else:
            distribution, vertex_features = self.backbone(bb_X, output_level="atom")

        if "vertex" in self.components:
            vertex_features = self.vertex_embedding(vertex_features)

        mvf = self.vertex_embedding.pooling(distribution, vertex_features)

        if return_embeddings:
            return mvf

        mvf = self.embedding(mvf)
        bias = None

        if "metaedge" in self.components:
            bias = self.metaedge_embedding(amds[:, self.metric_comp_size:])

        if "metavertex" in self.components:
            for layer in self.metavertex_embedding:
                mvf = self.dropout(layer(mvf, bias=bias))

        return self.out(mvf)


if __name__ == '__main__':
    num_heads = 3
    qdim = 6
    kdim = 6
    vdim = 1
    head_dim = 3
    prediction_dim = 5
    amd_k = 10

    # During inference
    q = torch.zeros(2, qdim)
    k = torch.zeros(6, kdim)
    v = torch.zeros(6, prediction_dim)
    initial_predictions = torch.zeros(2, prediction_dim)
    amds_target = torch.zeros(2, amd_k)
    amds_neighbors = torch.zeros(6, amd_k)
    bias = torch.cdist(amds_target, torch.concat([amds_neighbors, amds_target], dim=0))
    att = CrAKNEncoder(q.shape[1], k.shape[1], v.shape[1], num_heads, head_dim)
    master_k = torch.concat([k, q], dim=0)
    master_v = torch.concat([v, initial_predictions], dim=0)
    mask = torch.concat([torch.ones(q.shape[0], k.shape[0]), torch.eye(q.shape[0])], dim=1)
    att(q, master_k, master_v, bias=bias, mask=mask)

    # In training
    q = torch.ones(2, qdim)
    k = q
    v = torch.ones(q.shape[0], prediction_dim)
    initial_predictions = torch.zeros(2, prediction_dim)
    amds_target = torch.zeros(q.shape[0], amd_k)
    amds_neighbors = torch.zeros(k.shape[0], amd_k)
    amds = torch.concat([amds_neighbors, amds_target], dim=0)
    bias = torch.cdist(amds_target, amds)
    att = CrAKNEncoder(q.shape[1], k.shape[1], v.shape[1], num_heads, head_dim)
    master_k = torch.concat([k, q], dim=0)
    master_v = torch.concat([v, initial_predictions], dim=0)
    mask = torch.concat([-torch.eye(q.shape[0], k.shape[0]) + 1, torch.eye(q.shape[0])], dim=1)
    att(q, master_k, master_v, bias=bias)
