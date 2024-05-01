import math

import torch
from torch import nn
import torch.nn.functional as F

from crakn.backbones.pst_v2 import weighted_softmax


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class CrAKNVectorAttention2D(nn.Module):
    def __init__(
            self,
            embedding_dim,
            attention_dropout=0.0,
            qkv_bias=True,
            use_multiplier=True,
            use_bias=True,
            activation=nn.Mish
    ):
        super(CrAKNVectorAttention2D, self).__init__()

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

    def forward(self, feat, bias=None, mul=None):
        query, key, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )
        relation_qk = key.unsqueeze(-2) - query.unsqueeze(-1)
        if bias is not None:
            bias = bias.unsqueeze(-2) - bias.unsqueeze(-1)
            if mul is None:
                mul = bias
            else:
                mul = mul.unsqueeze(-2) - mul.unsqueeze(-1)
            if self.delta_mul:
                pem = self.linear_p_multiplier(mul)
                relation_qk = relation_qk * pem
            if self.delta_bias:
                peb = self.linear_p_bias(bias)
                relation_qk = relation_qk + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(F.softmax(weight, dim=-2))

        feat = torch.einsum("i j k, i k -> i k", weight, value)
        return feat


class CrAKNVectorAttention3D(nn.Module):
    def __init__(
            self,
            embed_channels,
            attention_dropout=0.0,
            qkv_bias=True,
            activation=nn.Mish
    ):
        super(CrAKNVectorAttention3D, self).__init__()

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            nn.LayerNorm(embed_channels),
            activation(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            nn.LayerNorm(embed_channels),
            activation(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, embed_channels),
            nn.LayerNorm(embed_channels),
            activation(inplace=True),
            nn.Linear(embed_channels, embed_channels),
        )

        self.softmax = nn.Softmax(dim=2)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, feat, distribution=None):
        if distribution is None:
            distribution = torch.sum(torch.abs(feat), dim=-1, keepdim=True)
            distribution[distribution > 0] = 1

        query, key, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )

        relation_qk = key.unsqueeze(-3) - query.unsqueeze(-2)

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(weighted_softmax(weight, dim=-2, weights=distribution.unsqueeze(1)))

        mask = (distribution * distribution.transpose(-1, -2)) > 0
        weight = weight * mask.unsqueeze(-1)
        feat = torch.einsum("b i j k, b j k -> b i k", weight, value)
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
