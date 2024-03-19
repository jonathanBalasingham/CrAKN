#! python
"""Module to train for a folder with formatted dataset."""
import csv
import os
import sys
import time
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson
import argparse
import glob
import torch
from typing import List
from pymatgen.core.structure import Structure
from .core.data import get_dataloader, CrAKNDataset

from .core.config import TrainingConfig
from .core.train import train_crakn

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

device = "cpu"

parser = argparse.ArgumentParser(
    description="Crystal Attribute Knowledge Network"
)
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    "--config_name",
    default="crakn/examples/sample_data/config_example.json",
    help="Name of the config file",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)


parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument(
    "--batch_size", default=None, help="Batch size, generally 64"
)

parser.add_argument(
    "--epochs", default=None, help="Number of epochs, generally 300"
)

parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)

parser.add_argument(
    "--device",
    default=None,
    help="set device for training the model [e.g. cpu, cuda, cuda:2]",
)

parser.add_argument(
    "--restart_model_path",
    default=None,
    help="Checkpoint file path for model",
)


def train_for_folder(
    root_dir="examples/sample_data",
    config_name="config.json",
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    restart_model_path=None,
    file_format="poscar",
    output_dir=None,
):
    """Train for a folder."""
    id_prop_dat = os.path.join(root_dir, "id_prop.csv")
    config = loadjson(config_name)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    # config.keep_data_order = keep_data_order
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    if restart_model_path is not None:
        print("Restarting model from:", restart_model_path)
        from .core.model import CrAKN, CrAKNConfig

        rest_config = loadjson(os.path.join(restart_model_path, "config.json"))
        print("rest_config", rest_config)
        model = CrAKN(CrAKNConfig(**rest_config["model"]))
        chk_glob = os.path.join(restart_model_path, "*.pt")
        tmp = "na"
        for i in glob.glob(chk_glob):
            tmp = i
        print("Checkpoint file", tmp)
        model.load_state_dict(torch.load(tmp, map_location=device)["model"])
        model.to(device)
    else:
        model = None
    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    dataset = []
    n_outputs = []
    structures: List[Structure] = []
    targets: List[float] = []
    ids = []
    multioutput = False
    lists_length_equal = True
    for i in data:
        info = {}
        file_name = i[0]
        file_path = os.path.join(root_dir, file_name)
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )

        structures.append(atoms.pymatgen_converter())

        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name
        ids.append(file_name)

        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        else:
            multioutput = True

        targets.append(tmp)
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset.append(info)

    if multioutput:
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]

    if multioutput and classification_threshold is not None:
        raise ValueError("Classification for multi-output not implemented.")
    if multioutput and lists_length_equal:
        config.backbone.output_features = len(n_outputs[0])
    else:
        # TODO: Pad with NaN
        if not lists_length_equal:
            raise ValueError("Make sure the outputs are of same size.")
        else:
            config.backbone.output_features = 1

    crakn_dataset = CrAKNDataset(structures, targets, ids, config)
    t1 = time.time()
    train_crakn(
        config,
        model,
        dataloaders=get_dataloader(crakn_dataset, config),
    )
    t2 = time.time()
    print("Time taken (s):", t2 - t1)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    train_for_folder(
        root_dir=args.root_dir,
        config_name=args.config_name,
        # keep_data_order=args.keep_data_order,
        classification_threshold=args.classification_threshold,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        file_format=args.file_format,
        restart_model_path=args.restart_model_path,
    )
