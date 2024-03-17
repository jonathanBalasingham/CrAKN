import torch
from torch import nn
import torch.nn.functional as F
import math
from crakn.utils import BaseSettings
from typing import Literal, Union
import dgl

from ..backbones.gcn import SimpleGCNConfig, SimpleGCN
from ..backbones.pst import PeriodicSetTransformer, PSTConfig


class CrAKNConfig(BaseSettings):
    name: Literal["crakn"] = "crakn"
    backbone: Literal["PST", "SimpleGCN"] = "PST"
    backbone_config: Union[PSTConfig, SimpleGCNConfig] = PSTConfig()
    embedding_dim: int = 64
    layers: int = 2
    num_heads: int = 4
    head_dim: int = 64
    dropout: float = 0
    output_features: int = 1
    amd_k: int = 100
    classification: bool = False


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


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class CrAKNAttention(nn.Module):

    def __init__(self, input_dim, num_heads, head_dim, dropout=0):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_heads * head_dim)
        self.embedding = nn.Linear(input_dim, num_heads * head_dim)
        self.bias_embedding = nn.Sequential(nn.Linear(input_dim, head_dim * num_heads),
                                            nn.Mish())
        self.diff_embedding = nn.Sequential(nn.Linear(head_dim * num_heads, head_dim * num_heads),
                                            nn.Mish())
        self.qkv_proj = nn.Linear(input_dim, 3 * num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, input_dim)
        self.bias_out = nn.Sequential(nn.Linear(head_dim * num_heads, input_dim),
                                      nn.Mish())
        self._reset_parameters()

    def _reset_parameters(self):
        #  From original torch implementation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, mask=None, bias=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)

        if bias is not None:
            attn_logits += bias

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x, mask=None, bias=None):
        seq_length = x.size()[0]
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        if bias is not None:
            bias = self.bias_embedding(bias)
            diffs = self.diff_embedding(bias[None, :, :] - bias[:, None, :])
            diffs = diffs.reshape(seq_length, seq_length, self.num_heads, self.head_dim)
            diffs = torch.norm(diffs, dim=-1)
            diffs = diffs.reshape(seq_length, seq_length, self.num_heads)
            diffs = diffs.permute(2, 0, 1)

        qkv = qkv.reshape(seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(1, 0, 2)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = self.scaled_dot_product(q, k, v, mask=mask, bias=diffs)
        values = values.permute(1, 0, 2)
        values = values.reshape(seq_length, self.num_heads * self.head_dim)
        values = self.dropout(values)
        return self.o_proj(values), self.bias_out(bias)


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

    def forward(self, g: dgl.DGLGraph, node_features: torch.Tensor, edge_features: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.dense(node_features))
        y = self.activation(self.dense_edge(edge_features))
        out = self.conv(g, x, y)
        return self.activation_out(self.proj_out(out))


class CrAKN(nn.Module):
    def __init__(self, config: CrAKNConfig):
        super().__init__()
        self.backbone = get_backbone(config.backbone, bb_config=config.backbone_config)
        self.retrieve_data = get_data_retriever(config.backbone)
        #self.layers = [CrAKNLayer(config.embedding_dim) for _ in range(config.layers)]
        self.embedding = nn.Linear(config.backbone_config.output_features, config.embedding_dim)
        self.layers = [CrAKNAttention(config.embedding_dim, config.num_heads, config.head_dim, config.dropout)
                       for _ in range(config.layers)]
        self.bias_embedding = nn.Linear(config.amd_k, config.embedding_dim)
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.out = nn.Linear(config.embedding_dim, config.output_features)
        self.bn = nn.BatchNorm1d(config.embedding_dim)

    def forward(self, inputs) -> torch.Tensor:
        backbone_input, amds, latt = inputs
        data = backbone_input[:-1]
        node_features = self.backbone(data)
        node_features = self.embedding(node_features)
        bias = self.bias_embedding(amds)

        x = self.ln1(node_features)
        for layer in self.layers:
            temp_x, temp_bias = layer(x, bias=bias)
            x = self.ln2(x + temp_x)
            bias = self.ln2(bias + temp_bias)

        return self.out(x)
