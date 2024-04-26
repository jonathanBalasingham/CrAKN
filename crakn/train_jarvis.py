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
    "--config",
    default="",
    help="Name of the config file",
)

parser.add_argument(
    "--target",
    default="",
    help="Target property"
)


def train_jarvis(config: TrainingConfig):
    history = train_crakn(config)


if __name__ == '__main__':
    import sys

    args = parser.parse_args(sys.argv[1:])
    print(f"Target: {args.target}")
    if args.config != "":
        config = loadjson(args.config)
        if args.target != "":
            config["target"] = args.target
    else:
        if args.target != "":
            config = TrainingConfig(target=args.target)
        else:
            config = TrainingConfig()
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    train_jarvis(config)
