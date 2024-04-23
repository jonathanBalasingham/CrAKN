import random

import torch
from torch import nn
import torch.nn.functional as F
import math

from crakn.backbones.matformer import MatformerConfig, Matformer
from crakn.backbones.pst_v2 import PeriodicSetTransformerV2, PSTv2Config
from crakn.backbones.utils import RBFExpansion
from crakn.utils import BaseSettings
from typing import Literal, Union, Tuple
import dgl

from ..backbones.gcn import SimpleGCNConfig, SimpleGCN
from ..backbones.pst import PeriodicSetTransformer, PSTConfig
from random import randrange


class CrAKNConfig(BaseSettings):
    name: Literal["crakn"]
    backbone: Literal["PST", "SimpleGCN", "Matformer", "PSTv2"] = "PST"
    mtype: Literal["Transformer", "GNN"] = "GNN"
    backbone_config: Union[PSTConfig, SimpleGCNConfig, MatformerConfig, PSTv2Config] = PSTConfig(name="PST")
    embedding_dim: int = 64
    layers: int = 4
    num_heads: int = 4
    head_dim: int = 128
    dropout: float = 0
    output_features: int = 1
    amd_k: int = 100
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
    else:
        raise NotImplementedError(f"Unknown backbone: {bb}")


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class CrAKNVectorAttention(nn.Module):
    def __init__(
            self,
            embedding_dim,
            attention_dropout=0.0,
            qkv_bias=True,
            use_multiplier=False,
            use_bias=False,
            activation=nn.Mish
    ):
        super(CrAKNVectorAttention, self).__init__()
        self.embed_channels = embedding_dim
        self.attn_drop_rate = attention_dropout
        self.qkv_bias = qkv_bias
        self.delta_mul = use_multiplier
        self.delta_bias = use_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias),
            nn.LayerNorm(embedding_dim),
            activation(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias),
            nn.LayerNorm(embedding_dim),
            activation(inplace=True),
        )

        self.linear_v = nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias)

        if self.delta_mul:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                activation(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
            )
        if self.delta_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                activation(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            activation(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.softmax = nn.Softmax(dim=2)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, feat, pos=None):
        query, key, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )
        relation_qk = key.unsqueeze(-2) - query.unsqueeze(-1)
        if pos is not None:
            pos = pos.unsqueeze(-2) - pos.unsqueeze(-1)
        if self.delta_mul:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.delta_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(F.softmax(weight, dim=-2))

        feat = torch.einsum("i j k, i k -> i k", weight, value)
        return feat


class CrAKNEncoder(nn.Module):

    def __init__(self, qdim, kdim, vdim, num_heads, head_dim, dropout=0, bias_dim=40, embed_value=False):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim
        self.bias_dim = bias_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_heads * head_dim)
        self.q_proj = nn.Linear(qdim, head_dim * num_heads)
        self.k_proj = nn.Linear(kdim, head_dim * num_heads)
        self.v_proj = nn.Linear(vdim, head_dim * num_heads)
        self.q_out = nn.Linear(head_dim * num_heads, qdim)
        self.k_out = nn.Linear(head_dim * num_heads, kdim)
        self.v_out = nn.Linear(head_dim * num_heads, vdim)
        self.q_ln = nn.LayerNorm(qdim)
        self.k_ln = nn.LayerNorm(kdim)
        self.v_ln = nn.LayerNorm(vdim)
        self.activation = nn.Mish()
        self.bias_proj = nn.Linear(bias_dim, num_heads * bias_dim)

        self.o_proj = nn.Sequential(nn.Linear(head_dim * num_heads, head_dim * num_heads),
                                    nn.Mish(), nn.Linear(head_dim * num_heads, vdim))
        self.bias_out = nn.Sequential(nn.Linear(bias_dim * num_heads, bias_dim),
                                      nn.Mish())
        self._reset_parameters()

    def _reset_linear_parameters(self, mod):
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            mod.bias.data.fill_(0)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
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

    def forward(self, q, k, v, mask=None, bias=None):
        q_shape = q.size()
        k_shape = k.size()
        v_shape = v.size()

        if mask is not None:
            mask = mask.unsqueeze(0)

        proj_q = self.q_proj(q).reshape(q.shape[0], self.num_heads, self.head_dim).permute(1, 0, 2)
        proj_k = self.k_proj(k).reshape(k.shape[0], self.num_heads, self.head_dim).permute(1, 0, 2)
        proj_v = v.unsqueeze(0).expand(self.num_heads, -1, -1)

        if bias is not None:
            bias = self.bias_proj(bias)
            diffs = torch.norm(
                bias.reshape(bias.shape[0], bias.shape[1], self.num_heads, self.bias_dim),
                dim=-1).permute(2, 0, 1)
        else:
            diffs = None

        values, attention = self.scaled_dot_product(proj_q, proj_k, proj_v, mask=mask, bias=diffs)

        values = values.permute(1, 0, 2).reshape(q_shape[0], self.vdim * self.num_heads)
        proj_q = proj_q.permute(1, 2, 0).reshape(q_shape[0], self.num_heads * self.head_dim)
        proj_k = proj_k.permute(1, 2, 0).reshape(k_shape[0], self.num_heads * self.head_dim)

        q = self.q_ln(q + self.activation(self.q_out(proj_q)))
        k = self.k_ln(k + self.activation(self.k_out(proj_k)))

        if bias is not None:
            return q, k, torch.mean(values, dim=-1, keepdim=True), self.bias_out(bias)
        else:
            return q, k, torch.mean(values, dim=-1, keepdim=True), bias


class CrAKN(nn.Module):
    def __init__(self, config: CrAKNConfig):
        super().__init__()
        self.backbone = get_backbone(config.backbone, bb_config=config.backbone_config)
        self.model_type = config.mtype
        # self.layers = [CrAKNLayer(config.embedding_dim) for _ in range(config.layers)]
        self.embedding = nn.Linear(config.backbone_config.output_features, config.embedding_dim)
        atom_feature_size = 200 if config.backbone_config.atom_features == "mat2vec" else 92
        self.edge_embedding = nn.Linear(config.amd_k + atom_feature_size, config.embedding_dim)
        self.expansion = RBFExpansion(0, config.cutoff, config.expansion_size)
        self.layers = nn.ModuleList(
            [CrAKNEncoder(config.embedding_dim, config.embedding_dim, 1,
                          config.num_heads, config.head_dim, dropout=config.dropout,
                          embed_value=False, bias_dim=config.expansion_size)
             for _ in range(config.layers)])
        self.node_updates = nn.ModuleList([CrAKNVectorAttention(config.embedding_dim) for _ in range(config.layers)])

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
        self.cutoff = config.cutoff
        self.crakn_out = nn.Sequential(nn.Linear(config.num_heads * config.embedding_dim, config.embedding_dim),
                                       nn.Mish(), nn.Linear(config.embedding_dim, 1))

        emb_dim = config.embedding_dim
        self.embedding = nn.Linear(
            config.backbone_config.output_features,
            emb_dim
        )

        self.diff_struct_embedding = nn.Sequential(
            nn.Linear(config.amd_k, emb_dim),
            nn.Mish()
        )

        self.diff_comp_embedding = nn.Sequential(
            nn.Linear(atom_feature_size, emb_dim),
            nn.Mish()
        )

        self.struct_conv_lin = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Mish()
        )

        self.comp_conv_lin = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Mish()
        )

        self.struct_conv = dgl.nn.GATv2Conv(
            emb_dim,
            emb_dim,
            num_heads=1,
        )

        self.comp_conv = dgl.nn.GATv2Conv(
            emb_dim,
            emb_dim,
            num_heads=1
        )

        self.struct_pna = nn.ModuleList([dgl.nn.PNAConv(
            emb_dim,
            emb_dim,
            ['mean', 'max', 'sum'],
            ['identity', 'amplification'],
            2.5,
            edge_feat_size=emb_dim
        ) for _ in range(config.layers)])

        self.comp_pna = nn.ModuleList([dgl.nn.PNAConv(
            emb_dim,
            emb_dim,
            ['mean', 'max', 'sum'],
            ['identity', 'amplification'],
            2.5,
            edge_feat_size=emb_dim
        ) for _ in range(config.layers)])

        self.struct_egat = nn.ModuleList([dgl.nn.pytorch.conv.EGATConv(
            in_node_feats=emb_dim,
            in_edge_feats=emb_dim,
            out_node_feats=emb_dim,
            out_edge_feats=emb_dim,
            num_heads=1
        ) for _ in range(config.layers)])

        self.comp_egat = nn.ModuleList([dgl.nn.pytorch.conv.EGATConv(
            in_node_feats=emb_dim,
            in_edge_feats=emb_dim,
            out_node_feats=emb_dim,
            out_edge_feats=emb_dim,
            num_heads=1
        ) for _ in range(config.layers)])

        self.struct_egnn = nn.ModuleList([dgl.nn.pytorch.conv.EGNNConv(
            emb_dim,
            emb_dim,
            emb_dim,
            emb_dim,
        ) for _ in range(config.layers)])

        self.comp_egnn = nn.ModuleList([dgl.nn.pytorch.conv.EGNNConv(
            emb_dim,
            emb_dim,
            emb_dim,
            emb_dim,
        ) for _ in range(config.layers)])

        self.ef_embedding = nn.Linear(config.backbone_config.outputs, emb_dim)

        self.bn = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(config.layers)]
        )

        self.out = nn.Linear(emb_dim, config.output_features)

    def transformer_forward(self, inputs, neighbors=None, direct=False) -> torch.Tensor:
        backbone_input, target_edge_features, latt, _ = inputs

        if direct:
            target_node_features = backbone_input[0]
        else:
            data = backbone_input[:-1]
            target_node_features = self.backbone(data)

        if neighbors is None:
            neighbor_node_features = target_node_features
            neighbor_edge_features = target_edge_features
            neighbor_latt = latt
            neighbor_target = backbone_input[-1]
            if neighbor_target.dim() == 1:
                neighbor_target = neighbor_target.unsqueeze(-1)
            num_nodes = neighbor_node_features.shape[0]
            mask = torch.concat([-torch.eye(num_nodes, device=target_node_features.device) + 1,
                                 torch.eye(num_nodes, device=target_node_features.device)], dim=1)
            mask.to(target_node_features.device)
        else:
            neighbor_node_features, neighbor_edge_features, neighbor_latt, neighbor_target = neighbors
            num_target_nodes = target_node_features.shape[0]
            num_neighbor_nodes = neighbor_node_features.shape[0]
            mask = torch.concat([torch.ones(num_target_nodes, num_neighbor_nodes, device=target_node_features.device),
                                 torch.eye(num_target_nodes, device=target_node_features.device)], dim=1)
            mask.to(target_node_features.device)

        if self.backbone_only:
            return self.out(target_node_features)

        predictions = self.backbone_out(target_node_features)

        node_features = self.embedding(target_node_features)
        neighbor_node_features = self.embedding(neighbor_node_features)

        q = node_features
        k = torch.concat([neighbor_node_features, node_features], dim=0)
        edge_features = torch.concat([neighbor_edge_features, target_edge_features], dim=0)

        if self.attention_bias:
            bias = torch.cdist(target_edge_features, edge_features).unsqueeze(-1)
            bias = self.expansion(bias).squeeze()
            edge_features = self.edge_embedding(edge_features)
        else:
            bias = None

        for layer, node_update in zip(self.layers, self.node_updates):
            q, k, predictions_updated, bias = layer(q, k, torch.concat([neighbor_target, predictions], dim=0),
                                                    bias=bias, mask=mask)
            k = node_update(k, pos=None)
            q = k[-q.shape[0]:]
            predictions = (predictions_updated + predictions) / 2

        return predictions

    def graph_forward(self, inputs):
        g, original_ids, target_ids = inputs
        g.local_var()
        node_features = g.ndata["node_features"]
        node_features = self.embedding(node_features)
        ef = self.ef_embedding(g.ndata["extra_features"])

        comp_edge_features = self.diff_comp_embedding(g.edata["comp"])
        struct_edge_features = self.diff_struct_embedding(g.edata["struct"])

        for c, s, bn in zip(self.comp_egnn, self.struct_egnn, self.bn):
            #node_features = c(g, node_features, edge_feat=comp_edge_features)
            #node_features = s(g, node_features, edge_feat=struct_edge_features)
            #node_features = bn(node_features)
            ef, node_features = c(g, ef, node_features, comp_edge_features)
            ef, node_features = s(g, ef, node_features, struct_edge_features)

        return self.out(node_features)[original_ids]

    def forward(self, inputs):
        if self.model_type == "Transformer":
            return self.transformer_forward(inputs)
        elif self.model_type == "GNN":
            return self.graph_forward(inputs)
        else:
            raise NotImplementedError(f"Model Type: {self.model_type} not implemented")


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
