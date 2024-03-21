import random
from functools import partial

import amd
import numpy as np
from tqdm import tqdm

from crakn.backbones.gcn import GCNData
from crakn.backbones.pst import PSTData
from crakn.config import TrainingConfig

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from typing import List, Tuple
from pymatgen.core.structure import Structure
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
# from matbench.bench import MatbenchBenchmark
import torch


def retrieve_data(config: TrainingConfig) -> Tuple[List[Structure], List[float], List]:
    if config.dataset == "matbench":
        pass
    else:
        d = data(config.dataset)

    structures: List[Structure] = []
    targets: List[float] = []
    ids = []
    current_id = 0
    for datum in tqdm(d, desc="Retrieving data.."):
        if config.target not in datum.keys():
            raise ValueError(f"Unknown target {config.target}")

        target = datum[config.target]
        if not isinstance(target, float):
            continue

        if config.id_tag in targets:
            ids.append(datum[config.id_tag])
        else:
            ids.append(current_id)
            current_id += 1
        atoms = (Atoms.from_dict(datum["atoms"]) if isinstance(datum["atoms"], dict) else datum["atoms"])
        structure = atoms.pymatgen_converter()
        structures.append(structure)
        targets.append(target)
    return structures, targets, ids


def get_dataset(structures, targets, config):
    if config.name == "PST":
        return PSTData(structures, targets, config)
    elif config.name == "SimpleGCN":
        return GCNData(structures, targets, config)
    else:
        raise NotImplementedError(f"Not implemented yet, {config.name}")


class CrAKNDataset(torch.utils.data.Dataset):

    def __init__(self, structures, targets, ids, config: TrainingConfig):
        super().__init__()
        self.data = get_dataset(structures, targets, config.base_config.backbone_config)

        periodic_sets = [amd.periodicset_from_pymatgen_structure(s) for s in
                         tqdm(structures, desc="Creating Periodic Sets..")]

        self.amds = np.vstack([amd.AMD(ps, k=config.base_config.amd_k)
                               for ps in tqdm(periodic_sets, desc="Calculating AMDs..")])
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

    test_sampler = SequentialSampler(indices[-test_size:])
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
