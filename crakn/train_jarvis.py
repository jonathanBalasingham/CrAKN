import torch
from crakn.config import TrainingConfig
from crakn.train import train_crakn
from jarvis.db.jsonutils import loadjson
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
    "--config_name",
    default="",
    help="Name of the config file",
)


def train_jarvis(config: TrainingConfig):
    history = train_crakn(config)


if __name__ == '__main__':
    import sys

    args = parser.parse_args(sys.argv[1:])
    if args.config_name != "":
        config = loadjson(args.config_name)
    else:
        config = TrainingConfig()
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    train_jarvis(config)
