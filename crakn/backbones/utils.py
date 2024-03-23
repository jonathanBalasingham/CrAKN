"""Shared model-building components."""
from typing import Optional

import numpy as np
import torch
from torch import nn
from pathlib import Path
import pandas as pd
import json


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )



class AtomFeaturizer(nn.Module):
    def __init__(self, id_prop_file="mat2vec.csv", use_cuda=True):
        super(AtomFeaturizer, self).__init__()
        path = Path(__file__).parent.parent / 'data' / id_prop_file

        if id_prop_file == "mat2vec.csv":
            af = pd.read_csv(path).to_numpy()[:, 1:].astype("float32")
            af = np.vstack([np.zeros(200), af, np.ones(200)])
        else:
            with open(path) as f:
                atom_fea = json.load(f)
            af = np.vstack([i for i in atom_fea.values()])
            af = np.vstack([np.zeros(92), af, np.ones(92)])  # last is the mask, first is for padding
        if use_cuda:
            self.atom_fea = torch.Tensor(af).cuda()
        else:
            self.atom_fea = torch.Tensor(af)

    def forward(self, x):
        return torch.squeeze(self.atom_fea[x.long()])


class DistanceExpansion(nn.Module):
    def __init__(self, size=5, use_cuda=True):
        super(DistanceExpansion, self).__init__()
        self.size = size
        if use_cuda:
            self.starter = torch.Tensor([i for i in range(size)]).cuda()
        else:
            self.starter = torch.Tensor([i for i in range(size)])
        self.starter /= size

    def forward(self, x):
        out = (1 - (x.flatten().reshape((-1, 1)) - self.starter)) ** 2
        return out.reshape((x.shape[0], x.shape[1], x.shape[2] * self.size))
