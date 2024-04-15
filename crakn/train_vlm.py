import time
from functools import partial, reduce

from typing import Any, Dict, Union, Tuple
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from crakn.core.model import CrAKN
from crakn.utils import Normalizer, mae, mse, rmse, AverageMeter
import pickle as pk
import numpy as np

from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn

from crakn.core.data import get_dataloader, retrieve_data, CrAKNDataset, prepare_crakn_batch
from crakn.config import TrainingConfig

from jarvis.db.jsonutils import dumpjson
import json
import pprint

import torch
from crakn.config import TrainingConfig
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

parser.add_argument(
    "--target",
    default="",
    help="Target property"
)


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    return optimizer


def train_crakn(
        config: Union[TrainingConfig, Dict[str, Any]],
        model: nn.Module = None,
        dataloaders: Tuple[DataLoader, DataLoader, DataLoader] = None,
        return_predictions: bool = False):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    import os

    if type(config) is dict:
        config = TrainingConfig(**config)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False

    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()

    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp)

    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        torch.cuda.manual_seed_all(config.random_seed)

    if dataloaders is None:
        structures, targets, ids = retrieve_data(config)
        dataset = CrAKNDataset(structures, targets, ids, config)
        train_loader, val_loader, test_loader = get_dataloader(dataset, config)
    else:
        train_loader, val_loader, test_loader = dataloaders

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    prepare_batch = partial(prepare_crakn_batch, device=device,
                            internal_prepare_batch=train_loader.dataset.data.prepare_batch,
                            variable=config.variable_batch_size)

    prepare_batch_test = partial(prepare_crakn_batch, device=device,
                                 internal_prepare_batch=train_loader.dataset.data.prepare_batch,
                                 variable=False)
    if classification:
        config.base_config.classification = True

    if model is None:
        net = CrAKN(config.base_config).backbone
    else:
        net = model

    net.to(device)
    optimizer = setup_optimizer(net.parameters(), config)

    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    elif config.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=0.1)

    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
    }
    criterion = criteria[config.criterion]

    # set up training engine and evaluators

    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    sample_targets = torch.concat([data[-1] for data in tqdm(train_loader, "Normalizing..")], dim=0)
    print(sample_targets.shape)
    normalizer = Normalizer(sample_targets)

    # train the model!
    for epoch in range(config.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        end = time.time()
        net.train()
        for step, dat in enumerate(train_loader):
            bb_data, amds, latt, ids, target = dat
            train_inputs = bb_data[0]
            if isinstance(train_inputs, list) or isinstance(train_inputs, tuple):
                train_inputs = [i.to(device) for i in train_inputs]
            else:
                train_inputs = train_inputs.to(device)
            target_normed = normalizer.norm(target).to(device)
            output = net(train_inputs, direct=True)
            loss = criterion(output, target_normed)
            prediction = normalizer.denorm(output.data.cpu())
            mae_error, mse_error, rmse_error = mae(prediction, target), mse(prediction, target), rmse(prediction,
                                                                                                      target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if epoch % 25 == 0 and step % 25 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, step, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
        scheduler.step()

    # Test the model

    net.eval()
    f = open(os.path.join(config.output_dir, "prediction_results_test_set.csv"), "w")
    f.write("id,target,prediction\n")
    targets = []
    predictions = []

    with torch.no_grad():
        for dat in test_loader:
            test_bb_data = dat[0][0]
            target = dat[-1]
            if isinstance(test_bb_data, list) or isinstance(test_bb_data, tuple):
                test_bb_data = [i.to(device) for i in test_bb_data]
            else:
                test_bb_data = [test_bb_data.to(device)]

            out_data = net(test_bb_data, direct=True).cpu().reshape(-1)
            target = target.cpu().numpy().flatten().tolist()

            if len(target) == 1:
                target = target[0]

            pred = normalizer.denorm(out_data).data.numpy().flatten().tolist()
            targets.append(target)
            predictions.append(pred)
        f.close()

        targets = reduce(lambda x, y: x + y, targets)
        predictions = reduce(lambda x, y: x + y, predictions)

        print("Test MAE:",
              mean_absolute_error(np.array(targets), np.array(predictions)))

        def mad(target):
            return torch.mean(torch.abs(target - torch.mean(target)))

        print(f"Test MAD: {mad(torch.Tensor(targets))}")

        with open("res.txt", "a") as f:
            f.write(
                f"Test MAE ({config.base_config.backbone}, {config.dataset}, {config.target}, {config.base_config.backbone_only}) :"
                f" {str(mean_absolute_error(np.array(targets), np.array(predictions)))} \n")

        resultsfile = os.path.join(
            config.output_dir, "prediction_results_test_set.csv"
        )

        model_file = os.path.join(config.output_dir,
                                  f"model_{config.base_config.backbone}_{config.dataset}_{config.target}.csv")

        torch.save(net, model_file)

        target_vals = np.array(targets, dtype="float").flatten()
        predictions = np.array(predictions, dtype="float").flatten()

        with open(resultsfile, "w") as f:
            print("target,prediction", file=f)
            for target_val, predicted_val in zip(target_vals, predictions):
                print(f"{target_val}, {predicted_val}", file=f)

    if return_predictions:
        return history, predictions
    return history


if __name__ == '__main__':
    import sys

    args = parser.parse_args(sys.argv[1:])
    print(f"Target: {args.target}")
    if args.config_name != "":
        config = loadjson(args.config_name)
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

    train_crakn(config)
