import random

import torch
from torch import nn
import torch.nn.functional as F
import math
from crakn.utils import BaseSettings
from typing import Literal, Union, Tuple
import dgl

from ..backbones.gcn import SimpleGCNConfig, SimpleGCN
from ..backbones.pst import PeriodicSetTransformer, PSTConfig
from random import randrange


class CrAKNConfig(BaseSettings):
    name: Literal["crakn"]
    backbone: Literal["PST", "SimpleGCN"] = "PST"
    backbone_config: Union[PSTConfig, SimpleGCNConfig] = PSTConfig(name="PST")
    embedding_dim: int = 64
    layers: int = 4
    num_heads: int = 4
    head_dim: int = 128
    dropout: float = 0
    output_features: int = 1
    amd_k: int = 100
    classification: bool = False
    backbone_only: bool = False
    attention_bias: bool = True
    embed_bias: bool = True
    cutoff: float = 1


def get_backbone(bb: str, bb_config) -> nn.Module:
    if bb == "PST":
        return PeriodicSetTransformer(bb_config)
    elif bb == "SimpleGCN":
        return SimpleGCN(bb_config)
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
        self.diff_embedding = nn.Sequential(nn.Linear(input_dim, head_dim * num_heads),
                                            nn.Mish())
        # self.qkv_proj = nn.Linear(input_dim, 3 * num_heads * head_dim)
        self.q_proj = nn.Linear(input_dim, head_dim * num_heads)
        self.k_proj = nn.Linear(input_dim, head_dim * num_heads)
        self.v_proj = nn.Linear(input_dim, head_dim * num_heads)
        self.o_proj = nn.Sequential(nn.Linear(head_dim * num_heads, head_dim * num_heads),
                                            nn.Mish(), nn.Linear(head_dim * num_heads, input_dim))
        self.bias_out = nn.Sequential(nn.Linear(head_dim * num_heads, input_dim),
                                      nn.Mish())
        self._reset_parameters()

    def _reset_linear_parameters(self, mod):
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            mod.bias.data.fill_(0)


    def _reset_parameters(self):
        #  From original torch implementation
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        #nn.init.xavier_uniform_(self.o_proj.weight)
        #self.o_proj.bias.data.fill_(0)
        #self.q_proj.apply(self._reset_linear_parameters)
        #self.k_proj.apply(self._reset_linear_parameters)
        #self.v_proj.apply(self._reset_linear_parameters)
        self.o_proj.apply(self._reset_linear_parameters)


    def scaled_dot_product(self, q, k, v, mask=None, bias=None):
        """
        k and v must have the same seq length
        """
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

    def forward(self, q, k, v, mask=None, bias=None, embed_bias=True, embed_value=False):
        embed_bias = embed_bias if bias is not None else False
        seq_length = k.shape[0]

        if mask is not None:
            mask = expand_mask(mask)

        q = self.q_proj(q).reshape(q.shape[0], self.num_heads, -1)
        k = self.k_proj(k).reshape(k.shape[0], self.num_heads, -1)
        if embed_value:
            v = self.v_proj(v)
        else:
            v = v.unsqueeze(1).expand(-1, self.num_heads, -1)
        v = v.reshape(v.shape[0], self.num_heads, -1)
        vdim = v.shape[-1]

        if bias is not None:
            # if embed_bias:
            #    bias = self.diff_embedding(bias)
            #    diffs = bias.reshape(q.shape[0], seq_length, self.num_heads, self.head_dim)
            #    diffs = torch.norm(diffs, dim=-1)
            # else:
            #    diffs = torch.norm(bias, dim=-1, keepdim=True)
            #    diffs = diffs.expand(-1, -1, self.num_heads)

            # diffs = bias.unsqueeze(-1).expand(-1, -1, self.num_heads)
            # diffs = diffs.reshape(q.shape[0], seq_length, self.num_heads)
            # diffs = diffs.permute(2, 0, 1)
            diffs = bias.unsqueeze(0).expand(self.num_heads, -1, -1)
        else:
            diffs = None

        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        values, attention = self.scaled_dot_product(q, k, v, mask=mask, bias=diffs)
        values = values.permute(1, 0, 2)

        if not embed_value:
            values = torch.mean(values, dim=1)
            if not embed_bias:
                return values, bias
            # return values, self.bias_out(bias)
            return values, bias

        values = values.reshape(seq_length, self.num_heads * self.head_dim)
        values = self.dropout(values)

        if bias is None:
            return self.o_proj(values), None
        else:
            if not embed_bias:
                return self.o_proj(values), bias
            # return self.o_proj(values), self.bias_out(bias)
            return self.o_proj(values), bias


class CrAKN(nn.Module):
    def __init__(self, config: CrAKNConfig):
        super().__init__()
        self.backbone = get_backbone(config.backbone, bb_config=config.backbone_config)
        # self.layers = [CrAKNLayer(config.embedding_dim) for _ in range(config.layers)]
        self.embedding = nn.Linear(config.backbone_config.output_features, config.embedding_dim)
        self.layers = nn.ModuleList(
            [CrAKNAttention(config.embedding_dim, config.num_heads, config.head_dim, config.dropout)
             for _ in range(config.layers)])
        self.layers2 = nn.ModuleList(
            [CrAKNAttention(config.embedding_dim, config.num_heads, config.head_dim, config.dropout)
             for _ in range(config.layers)])
        self.bias_embedding = nn.Linear(config.amd_k, config.embedding_dim)
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        if config.backbone_only:
            self.out = nn.Linear(config.backbone_config.output_features, config.output_features)
        else:
            self.out = nn.Linear(config.embedding_dim, config.output_features)
        self.backbone_out = nn.Linear(config.backbone_config.output_features, config.output_features)
        self.bn = nn.BatchNorm1d(config.embedding_dim)
        self.backbone_only = config.backbone_only
        self.attention_bias = config.attention_bias
        self.embed_bias = config.embed_bias

    def predict(self, target_nodes, source_nodes):
        target_backbone_inputs, target_amds, target_latt = target_nodes

    def forward(self, inputs, neighbors=None) -> torch.Tensor:
        backbone_input, edge_features, latt, _ = inputs

        data = backbone_input[:-1]
        node_features = self.backbone(data)
        if self.backbone_only:
            return self.out(node_features)

        predictions = self.backbone_out(node_features)
        node_features = self.embedding(node_features)

        if self.attention_bias:
            if self.embed_bias:
                edge_features = self.bias_embedding(edge_features)
                edge_features = self.bn(edge_features)
            bias = torch.cdist(edge_features, edge_features)
        else:
            bias = None

        x = self.ln1(node_features)
        for layer in self.layers2:
            x_temp, bias = layer(x, x, x, bias=bias, embed_bias=self.embed_bias, embed_value=True)
            x = self.ln2(x + x_temp)

        for layer, layer2 in zip(self.layers, self.layers2):
            predictions, bias = layer(x, x, predictions, bias=bias, embed_bias=self.embed_bias, embed_value=False)
            #x, bias = layer2(x, x, x, bias=bias, embed_bias=self.embed_bias, embed_value=True)

        return predictions


if __name__ == '__main__':
    num_heads = 3
    input_dim = 2
    head_dim = 3
    q = torch.zeros(2, input_dim)
    k = torch.zeros(6, input_dim)
    v = torch.zeros(6, 5)
    amds_target = torch.zeros(2, input_dim)
    amds_neighbors = torch.zeros(6, input_dim)
    bias = torch.cdist(amds_target, amds_neighbors)
    att = CrAKNAttention(input_dim, num_heads, head_dim)
    att(q, k, v, bias=bias)
