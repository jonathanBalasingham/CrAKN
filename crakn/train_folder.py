import csv
import os
import pickle

import torch
from tqdm import tqdm

from crakn.config import TrainingConfig
from crakn.core.data import DATA_FORMATS, CrAKNDataset, get_dataloader
from crakn.train import train_crakn
from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import Atoms
import argparse


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

parser = argparse.ArgumentParser(
    description="Crystal Attribute Knowledge Network"
)

parser.add_argument(
    "--config",
    default="",
    help="Name of the config file",
)

parser.add_argument(
    "--folder",
    default="",
    help="Location of CIF files"
)

parser.add_argument(
    "--cache",
    default=True,
    help="Write or read to cache",
)


def read_folder(folder_path: str, format: str):
    id_prop_file = os.path.join(folder_path, "id_prop.csv")
    with open(id_prop_file, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    files = os.listdir(folder_path)
    for file in files:
        ext = os.path.basename(file).split(".")[-1]
        if ext == "csv" or ext == "json":
            pass
        else:
            file_format = ext
            break

    print(f"Using file format: {file_format}")
    structures = []
    ids = []
    targets = []

    for i in tqdm(data, desc="Reading files from folders.."):
        file_name = i[0]
        if not file_name.endswith(file_format):
            file_name += f".{file_format}"
        file_path = os.path.join(folder_path, file_name)
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path, use_cif2cell=False)
        elif file_format == "xyz":
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )

        if format == "pymatgen":
            structure = atoms.pymatgen_converter()
            structures.append(structure)
        elif format == "jarvis":
            structures.append(atoms)

        ids.append(file_name)
        tmp = [float(j) for j in i[1:]]
        targets.append(tmp)

    return structures, targets, ids


def train_folder(config: TrainingConfig, folder_path: str, cache: bool):
    cache_path = os.path.join(folder_path, "cache_structures")
    dataloader_path = os.path.join(folder_path, "dataloaders")
    if os.path.exists(dataloader_path) and cache:
        print(f"Using cache file: {dataloader_path}")
        train_loader, val_loader, test_loader = pickle.load(open(dataloader_path, "rb"))
        history = train_crakn(config, dataloaders=(train_loader, val_loader, test_loader))
        return history
    else:
        structures, targets, ids = read_folder(folder_path, DATA_FORMATS[config.base_config.backbone])
        pickle.dump((structures, targets, ids), open(cache_path, "wb"))

    dataset = CrAKNDataset(structures, targets, ids, config)
    train_loader, val_loader, test_loader = get_dataloader(dataset, config)
    if not os.path.exists(dataloader_path):
        pickle.dump((train_loader, val_loader, test_loader), open(dataloader_path, "wb"))
    history = train_crakn(config, dataloaders=(train_loader, val_loader, test_loader))
    return history


if __name__ == '__main__':
    import sys

    args = parser.parse_args(sys.argv[1:])
    if args.config != "":
        config = loadjson(args.config)
    else:
        config = TrainingConfig()
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    train_folder(config, args.folder, cache=args.cache)
