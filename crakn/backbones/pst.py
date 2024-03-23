from typing import Literal, Tuple, List

import amd
import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
import math

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pathlib import Path

from ..utils import BaseSettings
from pydantic_settings import SettingsConfigDict
from .utils import DistanceExpansion, AtomFeaturizer


class PSTConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.gcn."""
    name: Literal["PST"]
    atom_input_features: int = 200
    encoders: int = 4
    num_heads: int = 4
    embedding_features: int = 128
    output_features: int = 1
    dropout: float = 0
    attention_dropout: float = 0
    use_cuda: bool = torch.cuda.is_available()
    decoder_layers: int = 2
    expansion_size: int = 10
    k: int = 15
    collapse_tol: float = 1e-4
    atom_encoding: str = "mat2vec"
    model_config = SettingsConfigDict(env_prefix="jv_model")


def weighted_softmax(x, dim=-1, weights=None):
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    if weights is not None:
        x_exp = weights * x_exp
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    probs = x_exp / x_exp_sum
    return probs


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MHA(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.0, use_kv_bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.use_kv_bias = use_kv_bias
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.delta_mul = nn.Linear(embed_dim, embed_dim)
        self.delta_bias = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        #  From original torch implementation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, mask=None, weights=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = weighted_softmax(attn_logits, dim=-1, weights=weights)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x, mask=None, return_attention=False, weights=None, bias=None):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        if weights is not None:
            weights = torch.transpose(weights, -2, -1)
            weights = weights[:, None, :, :].expand(-1, self.num_heads, -1, -1)

        values, attention = self.scaled_dot_product(q, k, v, mask=mask, weights=weights)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        values = self.dropout(values)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class PeriodicSetTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, attention_dropout=0.0, dropout=0.0, activation=nn.Mish):
        super(PeriodicSetTransformerEncoder, self).__init__()
        self.embedding = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.out = nn.Linear(embedding_dim * num_heads, embedding_dim)
        self.multihead_attention = MHA(embedding_dim, embedding_dim * num_heads, num_heads, dropout=attention_dropout)
        self.pre_norm = nn.LayerNorm(embedding_dim)
        self.ln = torch.nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.W_q = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.W_k = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.W_v = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.ffn = nn.Linear(embedding_dim, embedding_dim)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                 activation())

    def forward(self, x, weights, use_weights=True):
        x_norm = self.ln(x)
        keep = weights > 0
        keep = keep * torch.transpose(keep, -2, -1)

        if use_weights:
            att_output = self.multihead_attention(x_norm, weights=weights, mask=keep)
        else:
            att_output = self.multihead_attention(x_norm, mask=keep)

        output1 = x + self.out(att_output)
        output2 = self.ln(output1)
        output2 = self.ffn(output2)
        return self.ln(output1 + output2)


class PeriodicSetTransformer(nn.Module):

    def __init__(self, config: PSTConfig = PSTConfig(name="PST")):
        super(PeriodicSetTransformer, self).__init__()

        if config.atom_encoding not in ["mat2vec", "cgcnn"]:
            raise ValueError(f"atom_encoding_dim must be in {['mat2vec', 'cgcnn']}")
        else:
            atom_encoding_dim = 200 if config.atom_encoding == "mat2vec" else 92
            id_prop_file = "mat2vec.csv" if config.atom_encoding == "mat2vec" else "atom_init.json"

        self.pdd_embedding_layer = nn.Linear(config.k * config.expansion_size, config.embedding_features)
        self.comp_embedding_layer = nn.Linear(atom_encoding_dim, config.embedding_features)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.af = AtomFeaturizer(use_cuda=config.use_cuda, id_prop_file=id_prop_file)
        self.de = DistanceExpansion(size=config.expansion_size, use_cuda=config.use_cuda)
        self.ln = nn.LayerNorm(config.embedding_features)
        self.ln2 = nn.LayerNorm(config.embedding_features)

        self.encoders = nn.ModuleList(
            [PeriodicSetTransformerEncoder(config.embedding_features, config.num_heads, attention_dropout=config.attention_dropout,
                                           activation=nn.Mish) for _ in
             range(config.encoders)])
        self.decoder = nn.ModuleList([nn.Linear(config.embedding_features, config.embedding_features)
                                      for _ in range(config.decoder_layers - 1)])
        self.activations = nn.ModuleList([nn.Mish()
                                          for _ in range(config.decoder_layers - 1)])
        self.out = nn.Linear(config.embedding_features, config.output_features)
        print(f"Output of PST will have dimension: {config.output_features}")

    def forward(self, features):
        str_fea, comp_fea = features
        weights = str_fea[:, :, 0, None]
        comp_features = self.af(comp_fea)
        comp_features = self.comp_embedding_layer(comp_features)
        comp_features = self.dropout_layer(comp_features)
        str_features = str_fea[:, :, 1:]
        str_features = self.pdd_embedding_layer(self.de(str_features))

        x = comp_features + str_features

        x_init = x
        for encoder in self.encoders:
            x = encoder(x, weights)

        x = torch.sum(weights * (x + x_init), dim=1)

        x = self.ln2(x)
        for layer, activation in zip(self.decoder, self.activations):
            x = layer(x)
            x = activation(x)

        return self.out(x)


def preprocess_pdds(pdds_):
    min_pdd = np.min(np.vstack([np.min(pdd, axis=0) for pdd in pdds_]), axis=0)
    max_pdd = np.max(np.vstack([np.max(pdd, axis=0) for pdd in pdds_]), axis=0)
    pdds = [np.hstack(
        [pdd[:, 0, None], (pdd[:, 1:] - min_pdd[1:]) / (max_pdd[1:] - min_pdd[1:])]) for
        pdd
        in pdds_]
    return pdds


class PSTData(torch.utils.data.Dataset):
    def __init__(self, structures, targets, config: PSTConfig):
        self.k = int(config.k)
        self.collapse_tol = float(config.collapse_tol)
        self.id_prop_data = targets
        pdds = []
        periodic_sets = [amd.periodicset_from_pymatgen_structure(s) for s in structures]
        atom_fea = []
        for i in tqdm(range(len(periodic_sets)),
                      desc="Creating PDDs…",
                      ascii=False, ncols=75):
            ps = periodic_sets[i]
            pdd = amd.PDD(ps, k=self.k, collapse=False, collapse_tol=self.collapse_tol)
            atom_features = ps.types[ps.asym_unit][:, None]
            atom_fea.append(atom_features)
            pdds.append(pdd)

        self.pdds = preprocess_pdds(pdds)
        self.atom_fea = atom_fea

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx], self.id_prop_data[idx]
        return torch.Tensor(self.pdds[idx]), \
            torch.Tensor(self.atom_fea[idx]), \
            torch.Tensor([float(target)]), \
            cif_id

    @staticmethod
    def collate_fn(dataset_list):
        batch_fea = []
        composition_fea = []
        batch_target = []
        batch_cif_ids = []

        for i, (structure_features, comp_features, target, cif_id) in enumerate(dataset_list):
            batch_fea.append(structure_features)
            composition_fea.append(comp_features)
            batch_target.append(target)
            batch_cif_ids.append(cif_id)

        return (pad_sequence(batch_fea, batch_first=True),
                pad_sequence(composition_fea, batch_first=True)), \
            torch.stack(batch_target, dim=0), \
            batch_cif_ids

    @staticmethod
    def prepare_batch(
            batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, List],
            device=None, non_blocking=False, subset=None
    ):
        """Send batched dgl crystal graph to device."""
        (pdd, comp), t, _ = batch
        if subset is None:
            subset = len(t)
        batch = (
            pdd[:subset].to(device, non_blocking=non_blocking),
            comp[:subset].to(device, non_blocking=non_blocking),
            t[:subset].to(device, non_blocking=non_blocking)
        )
        return batch


if __name__ == '__main__':
    from crakn.core.data import retrieve_data
    from crakn.config import TrainingConfig
    config = TrainingConfig()
    structures, targets, ids = retrieve_data(config)
    pst_dataset = PSTData(structures, targets, config.backbone)
    pst = PeriodicSetTransformer()
