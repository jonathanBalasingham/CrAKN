import random
from functools import partial, reduce
from math import nan
from pathlib import Path

import amd
import jarvis.core.atoms
import numpy as np
from tqdm import tqdm

from crakn.backbones.gcn import GCNData
from crakn.backbones.matformer import MatformerData
from crakn.backbones.pst import PSTData
from crakn.backbones.pst_v2 import PSTv2Data
from crakn.config import TrainingConfig

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from typing import List, Tuple
from pymatgen.core.structure import Structure
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
import pandas as pd
# from matbench.bench import MatbenchBenchmark
import torch

from crakn.core.graph import knowledge_graph

DATA_FORMATS = {
    "PST": "pymatgen",
    "Matformer": "jarvis",
    "SimpleGCN": "pymatgen",
    "PSTv2": "pymatgen"
}


def retrieve_data(config: TrainingConfig) -> Tuple[List[Structure], List[List[float]], List]:
    if config.dataset == "matbench":
        pass
    else:
        d = data(config.dataset)

    structures: List[Structure] = []
    targets: List[List[float]] = []
    ids = []
    current_id = 0
    for datum in tqdm(d, desc="Retrieving data.."):
        for t in config.target:
            if t not in datum.keys():
                raise ValueError(f"Unknown target {config.target}")

        target = []
        for property_name in config.target:
            val = datum[property_name]
            if isinstance(val, float):
                target.append(val)
            else:
                target.append(nan)

        if np.all(np.isnan(target)):
            continue

        if config.id_tag in datum:
            ids.append(datum[config.id_tag])
        else:
            ids.append(current_id)
            current_id += 1
        atoms = (Atoms.from_dict(datum["atoms"]) if isinstance(datum["atoms"], dict) else datum["atoms"])
        if DATA_FORMATS[config.base_config.backbone] == "pymatgen":
            structure = atoms.pymatgen_converter()
            structures.append(structure)
        elif DATA_FORMATS[config.base_config.backbone] == "jarvis":
            structures.append(atoms)
        targets.append(target)
    return structures, targets, ids


def get_dataset(structures, targets, ids, config):
    if config.name == "PST":
        return PSTData(structures, targets, ids, config)
    elif config.name == "PSTv2":
        return PSTv2Data(structures, targets, ids, config)
    elif config.name == "SimpleGCN":
        return GCNData(structures, targets, ids, config)
    elif config.name == "Matformer":
        return MatformerData(structures, targets, ids, config)
    else:
        raise NotImplementedError(f"Not implemented yet, {config.name}")


class CrAKNDataset(torch.utils.data.Dataset):

    def __init__(self, structures, targets, ids, config: TrainingConfig):
        super().__init__()
        self.data = get_dataset(structures, targets, ids, config.base_config.backbone_config)

        if isinstance(structures[0], jarvis.core.atoms.Atoms):
            structures = [s.pymatgen_converter() for s in structures]
        periodic_sets = [amd.periodicset_from_pymatgen_structure(s) for s in
                         tqdm(structures, desc="Creating Periodic Sets..")]
        mat2vec = pd.read_csv(Path(__file__).parent.parent / "data" / "mat2vec.csv").to_numpy()[:, 1:].astype(
            np.float64)
        comp = np.vstack([np.mean(mat2vec[ps.types - 1], axis=0) for ps in periodic_sets])
        amds = np.vstack([amd.AMD(ps, k=config.base_config.amd_k)
                          for ps in tqdm(periodic_sets, desc="Calculating AMDs..")])
        self.amds = np.concatenate([comp, amds], axis=1)
        self.lattices = np.array([list(s.lattice.parameters)
                                  for s in tqdm(structures, desc="Retrieving lattices..")])
        self.targets = targets
        self.ids = ids

    def __len__(self):
        """Get length."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        return (self.data[idx], torch.Tensor(self.amds[idx]), torch.Tensor(self.lattices[idx]),
                self.ids[idx], torch.Tensor(self.targets[idx]))


def _convert(vlm: torch.nn.Module, loader: torch.utils.data.DataLoader, target_index, original_data=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    node_features = []
    AMDs = []
    lattices = []
    targets = []
    cids = []
    with torch.no_grad():
        for datum in loader:
            bb_data, amds, latt, ids, target = datum
            inds = np.where(np.logical_not(np.isnan(target[:, target_index])))[0]
            if inds.shape == target.shape[0]:
                continue

            amds = amds[inds]
            latt = latt[inds]
            ids = [ids[i] for i in inds]
            target = target[inds]
            train_inputs = bb_data[0]

            if isinstance(train_inputs, list) or isinstance(train_inputs, tuple):
                train_inputs = [i.to(device) for i in train_inputs]
            else:
                train_inputs = train_inputs.to(device)

            nf = vlm(train_inputs)
            base_preds = vlm(train_inputs, output_level="property")
            nf = torch.hstack([nf[inds], base_preds[inds]])
            node_features.append(nf)
            lattices.append(latt)
            AMDs.append(amds)
            targets.append(target[:, target_index])
            cids.append(ids)

    node_features = torch.concat(node_features, dim=0)
    AMDs = torch.concat(AMDs, dim=0)
    lattices = torch.concat(lattices, dim=0)
    targets = torch.concat(targets, dim=0)
    cids = reduce(lambda x, y: x + y, cids)

    return PretrainCrAKNDataset(node_features, AMDs, lattices, targets, cids)


def convert_to_pretrain_dataset(vlm: torch.nn.Module,
                                train_loader: torch.utils.data.DataLoader,
                                val_loader: torch.utils.data.DataLoader,
                                test_loader: torch.utils.data.DataLoader,
                                config: TrainingConfig):
    d = retrieve_data(config)
    train_dataset = _convert(vlm, train_loader, target_index=config.mo_target_index, original_data=d)
    val_dataset = _convert(vlm, val_loader, target_index=config.mo_target_index, original_data=d)
    test_dataset = _convert(vlm, test_loader, target_index=config.mo_target_index, original_data=d)

    if config.base_config.mtype == "GNN":
        return knowledge_graph(train_dataset, val_dataset, test_dataset, config)

    return (DataLoader(dataset, batch_size=config.batch_size,
                       sampler=SubsetRandomSampler([i for i in range(len(dataset))]),
                       num_workers=config.num_workers,
                       collate_fn=collate_pretrain_crakn_data, pin_memory=config.pin_memory,
                       shuffle=False) for dataset in [train_dataset, val_dataset, test_dataset])


class PretrainCrAKNDataset(torch.utils.data.Dataset):

    def __init__(self, vlm_output, amds, lattices, targets, ids):
        super().__init__()
        self.data = vlm_output
        self.amds = amds
        self.lattices = lattices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        """Get length."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        return (self.data[idx], torch.Tensor(self.amds[idx]), torch.Tensor(self.lattices[idx]),
                self.ids[idx], torch.Tensor([self.targets[idx]]))


def collate_crakn_data(dataset_list, internal_collate=PSTData.collate_fn):
    backbone_data = []
    lattices = []
    amds = []
    targets = []
    ids = []

    for bd, amd_fea, latt, datum_id, target in dataset_list:
        backbone_data.append(bd)
        amds.append(amd_fea)
        lattices.append(latt)
        targets.append(target)
        ids.append(datum_id)

    return (internal_collate(backbone_data),
            torch.stack(amds, dim=0),
            torch.stack(lattices, dim=0),
            ids,
            torch.stack(targets, dim=0))


def collate_pretrain_crakn_data(dataset_list):
    node_features = []
    lattices = []
    amds = []
    targets = []
    ids = []

    for nf, amd_fea, latt, datum_id, target in dataset_list:
        node_features.append(nf)
        amds.append(amd_fea)
        lattices.append(latt)
        targets.append(target)
        ids.append(datum_id)

    return (torch.stack(node_features, dim=0),
            torch.stack(amds, dim=0),
            torch.stack(lattices, dim=0),
            ids,
            torch.stack(targets, dim=0))


def prepare_crakn_batch(batch, device=None, internal_prepare_batch=None, non_blocking=False, variable=True):
    if variable:
        subset = random.randrange(len(batch[-1]) // 4, len(batch[-1]))
    else:
        subset = len(batch[-1])

    batch = (
        (internal_prepare_batch(batch[0], device=device, non_blocking=non_blocking, subset=subset),
         batch[1][:subset].to(device=device, non_blocking=non_blocking),
         batch[2][:subset].to(device=device, non_blocking=non_blocking),
         batch[3][:subset]),
        batch[4][:subset].to(device=device, non_blocking=non_blocking),
    )
    return batch


def get_dataloader(dataset: CrAKNDataset, config: TrainingConfig):
    total_size = len(dataset)
    random.seed(config.random_seed)
    if config.n_train is None:
        if config.train_ratio is None:
            assert config.val_ratio + config.test_ratio < 1
            train_ratio = 1 - config.val_ratio - config.test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert config.train_ratio + config.val_ratio + config.test_ratio <= 1
            train_ratio = config.train_ratio
    indices = list(range(total_size))
    if not config.keep_data_order:
        random.shuffle(indices)

    if config.n_train:
        train_size = config.n_train
    else:
        train_size = int(train_ratio * total_size)
    if config.n_test:
        test_size = config.n_test
    else:
        test_size = int(config.test_ratio * total_size)
    if config.n_val:
        valid_size = config.n_val
    else:
        valid_size = int(config.val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])

    collate_fn = partial(collate_crakn_data, internal_collate=dataset.data.collate_fn)

    train_loader = DataLoader(dataset, batch_size=config.batch_size,
                              sampler=train_sampler,
                              num_workers=config.num_workers,
                              collate_fn=collate_fn, pin_memory=config.pin_memory,
                              shuffle=False)
    val_loader = DataLoader(dataset, batch_size=config.batch_size,
                            sampler=val_sampler,
                            num_workers=config.num_workers,
                            collate_fn=collate_fn, pin_memory=config.pin_memory,
                            shuffle=False)

    """test_loader = DataLoader(dataset, batch_size=config.batch_size,
                             sampler=test_sampler,
                             num_workers=config.num_workers,
                             collate_fn=collate_fn, pin_memory=config.pin_memory)"""
    test_set = torch.utils.data.Subset(dataset, indices[-test_size:])
    test_loader = DataLoader(test_set, batch_size=config.test_batch_size,
                             # sampler=test_sampler,
                             num_workers=config.num_workers,
                             collate_fn=collate_fn, pin_memory=config.pin_memory)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    fake_config = TrainingConfig()
    structures, targets, ids = retrieve_data(fake_config)
    dataset = CrAKNDataset(structures, targets, ids, fake_config)
