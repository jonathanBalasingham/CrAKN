from typing import Literal, Tuple, List

import amd
import torch
import torch.nn as nn
import numpy as np
import math

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from ..utils import BaseSettings
from pydantic_settings import SettingsConfigDict
from .utils import DistanceExpansion, AtomFeaturizer, RBFExpansion
from .pdd_helpers import custom_PDD, get_relative_vectors


class PSTv2Config(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.gcn."""
    name: Literal["PSTv2"]
    atom_input_features: int = 200
    encoders: int = 4
    num_heads: int = 1
    embedding_features: int = 128
    output_features: int = 1
    dropout: float = 0
    attention_dropout: float = 0
    use_cuda: bool = torch.cuda.is_available()
    decoder_layers: int = 2
    expansion_size: int = 10
    bias_expansion: int = 40
    k: int = 15
    collapse_tol: float = 1e-4
    atom_features: str = "mat2vec"
    outputs: int = 1
    model_config = SettingsConfigDict(env_prefix="jv_model")


class MLP(nn.Module):

    def __init__(self, input_dim: int, embedding_dim: int, layers: int, activation):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.net = nn.ModuleList(
            [nn.Sequential(nn.Linear(embedding_dim, embedding_dim), activation()) for _ in range(layers - 1)])

    def forward(self, x):
        x = self.embedding(x)
        for perceptron in self.net:
            x = perceptron(x)
        return x


def weighted_softmax(x, dim=-1, weights=None):
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    if weights is not None:
        x_exp = weights * x_exp
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    probs = x_exp / x_exp_sum
    return probs


class VectorAttention(nn.Module):
    def __init__(
            self,
            embed_channels,
            attention_dropout=0.0,
            qkv_bias=True,
            use_multiplier=False,
            use_bias=False,
            activation=nn.Mish
    ):
        super(VectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.attn_drop_rate = attention_dropout
        self.qkv_bias = qkv_bias
        self.delta_mul = use_multiplier
        self.delta_bias = use_bias

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

        if self.delta_mul:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(embed_channels, embed_channels),
                nn.LayerNorm(embed_channels),
                activation(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.delta_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(embed_channels, embed_channels),
                nn.LayerNorm(embed_channels),
                activation(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, embed_channels),
            nn.LayerNorm(embed_channels),
            activation(inplace=True),
            nn.Linear(embed_channels, embed_channels),
        )
        self.softmax = nn.Softmax(dim=2)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, feat, pos, distribution):
        query, key, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )
        relation_qk = key.unsqueeze(-3) - query.unsqueeze(-2)
        if self.delta_mul:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.delta_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(weighted_softmax(weight, dim=-2, weights=distribution.unsqueeze(1)))

        mask = (distribution * distribution.transpose(-1, -2)) > 0
        weight = weight * mask.unsqueeze(-1)
        feat = torch.einsum("b i j k, b j k -> b i k", weight, value)
        return feat


class PeriodicSetTransformerV2Encoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, attention_dropout=0.0, dropout=0.0, activation=nn.Mish):
        super(PeriodicSetTransformerV2Encoder, self).__init__()
        self.embedding = nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.out = nn.Linear(embedding_dim * num_heads, embedding_dim)
        self.vector_attention = VectorAttention(embedding_dim, attention_dropout)
        self.pre_norm = nn.LayerNorm(embedding_dim)
        self.ln = torch.nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                 activation())

    def forward(self, x, distribution, pos):
        x_norm = self.ln(x)
        att_output = self.vector_attention(x_norm, pos, distribution)
        output1 = x + self.out(att_output)
        output2 = self.ln(output1)
        output2 = self.ffn(output2)
        return self.ln(output1 + output2)


class PeriodicSetTransformerV2(nn.Module):

    def __init__(self, config: PSTv2Config = PSTv2Config(name="PSTv2")):
        super(PeriodicSetTransformerV2, self).__init__()

        if config.atom_features not in ["mat2vec", "cgcnn"]:
            raise ValueError(f"atom_encoding_dim must be in {['mat2vec', 'cgcnn']}")
        else:
            atom_encoding_dim = 200 if config.atom_features == "mat2vec" else 92
            id_prop_file = "mat2vec.csv" if config.atom_features == "mat2vec" else "atom_init.json"

        self.pdd_embedding_layer = nn.Linear(config.k * config.expansion_size, config.embedding_features)
        self.comp_embedding_layer = nn.Linear(atom_encoding_dim, config.embedding_features)
        self.af = AtomFeaturizer(use_cuda=config.use_cuda, id_prop_file=id_prop_file)
        self.de = DistanceExpansion(size=config.expansion_size, use_cuda=config.use_cuda)
        self.ln = nn.LayerNorm(config.embedding_features)
        self.rbf = RBFExpansion(bins=config.bias_expansion)
        self.pos_embedding = nn.Linear(config.bias_expansion, config.embedding_features)

        self.encoders = nn.ModuleList(
            [PeriodicSetTransformerV2Encoder(config.embedding_features, config.num_heads,
                                             attention_dropout=config.attention_dropout,
                                             activation=nn.Mish) for _ in
             range(config.encoders)])
        self.decoder = MLP(config.embedding_features, config.embedding_features,
                           config.decoder_layers, nn.Mish)
        self.out = nn.Linear(config.embedding_features, config.output_features)
        self.final = nn.Linear(config.output_features, config.outputs)

    def forward(self, features, output_level: Literal["atom", "crystal", "property"] = "crystal"):
        str_fea, comp_fea, cloud_fea = features
        distribution = str_fea[:, :, 0, None]
        str_features = str_fea[:, :, 1:]

        comp_features = self.comp_embedding_layer(self.af(comp_fea))
        str_features = self.pdd_embedding_layer(self.de(str_features))

        x = comp_features + str_features
        pos = cloud_fea.unsqueeze(1) - cloud_fea.unsqueeze(2)
        pos = torch.norm(pos, dim=-1, keepdim=True)
        pos = self.rbf(pos).squeeze()
        pos = self.pos_embedding(pos)

        x_init = x
        for encoder in self.encoders:
            x = encoder(x, distribution, pos)

        if output_level == "atom":
            return torch.concatenate([distribution, x], dim=-1)

        x = torch.sum(distribution * (x + x_init), dim=1)

        if output_level == "crystal":
            return x

        x = self.decoder(self.ln(x))
        x = self.out(x)
        return self.final(x)


def preprocess_pdds(pdds_):
    min_pdd = np.min(np.vstack([np.min(pdd, axis=0) for pdd in pdds_]), axis=0)
    max_pdd = np.max(np.vstack([np.max(pdd, axis=0) for pdd in pdds_]), axis=0)
    pdds = [np.hstack(
        [pdd[:, 0, None], (pdd[:, 1:] - min_pdd[1:]) / (max_pdd[1:] - min_pdd[1:])]) for
        pdd
        in pdds_]
    return pdds


class PSTv2Data(torch.utils.data.Dataset):
    def __init__(self, structures, targets, config: PSTv2Config):
        self.k = int(config.k)
        self.collapse_tol = float(config.collapse_tol)
        self.id_prop_data = targets
        pdds = []
        periodic_sets = [amd.periodicset_from_pymatgen_structure(s) for s in structures]
        atom_fea = []
        clouds = []
        for ind in tqdm(range(len(periodic_sets)),
                        desc="Creating PDDs…",
                        ascii=False, ncols=75):
            ps = periodic_sets[ind]
            pdd, groups, inds, cloud = custom_PDD(ps, k=self.k, collapse=True, collapse_tol=self.collapse_tol,
                                                  constrained=True, lexsort=False)
            indices_in_graph = [i[0] for i in groups]
            atom_features = ps.types[indices_in_graph][:, None]
            atom_fea.append(atom_features)
            pdds.append(pdd)
            points = cloud[[i[0] for i in groups]]
            clouds.append(points)

        self.pdds = preprocess_pdds(pdds)
        self.atom_fea = atom_fea
        self.clouds = clouds

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx], self.id_prop_data[idx]
        return torch.Tensor(self.pdds[idx]), \
            torch.Tensor(self.atom_fea[idx]), \
            torch.Tensor(self.clouds[idx]), \
            torch.Tensor(target), \
            cif_id

    @staticmethod
    def collate_fn(dataset_list):
        batch_fea = []
        composition_fea = []
        batch_clouds = []
        batch_target = []
        batch_cif_ids = []

        for i, (structure_features, comp_features, cloud_fea, target, cif_id) in enumerate(dataset_list):
            batch_fea.append(structure_features)
            composition_fea.append(comp_features)
            batch_clouds.append(cloud_fea)
            batch_target.append(target)
            batch_cif_ids.append(cif_id)

        return (pad_sequence(batch_fea, batch_first=True),
                pad_sequence(composition_fea, batch_first=True),
                pad_sequence(batch_clouds, batch_first=True)), \
            torch.stack(batch_target, dim=0), \
            batch_cif_ids

    @staticmethod
    def prepare_batch(
            batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, List],
            device=None, non_blocking=False, subset=None
    ):
        """Send batched dgl crystal graph to device."""
        (pdd, comp, cloud), t, _ = batch
        if subset is None:
            subset = len(t)
        batch = (
            pdd[:subset].to(device, non_blocking=non_blocking),
            comp[:subset].to(device, non_blocking=non_blocking),
            cloud[:subset].to(device, non_blocking=non_blocking),
            t[:subset].to(device, non_blocking=non_blocking)
        )
        return batch


if __name__ == '__main__':
    from crakn.core.data import retrieve_data
    from crakn.config import TrainingConfig

    config = TrainingConfig()
    structures, targets, ids = retrieve_data(config)
    pst_dataset = PSTv2Data(structures, targets, config.backbone)
    pst = PeriodicSetTransformerV2()
