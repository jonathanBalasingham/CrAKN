import pickle
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

import json
import pprint

import torch
from crakn.config import TrainingConfig
from crakn.train import setup_optimizer
from jarvis.db.jsonutils import loadjson, dumpjson
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


def train_vlm(
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

    output_directory = f"{config.base_config.backbone}_{config.dataset}_{config.target}"
    output_path = os.path.join(config.output_dir, output_directory)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    checkpoint_dir = os.path.join(output_path)
    deterministic = False
    classification = False

    tmp = config.dict()
    f = open(os.path.join(output_path, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()

    global tmp_output_dir
    tmp_output_dir = output_path
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

    dataloader_savepath = os.path.join(output_path,
                                       f"dataloader_{config.base_config.backbone}_{config.dataset}_{config.target}")
    with open(dataloader_savepath, "wb") as f:
        pickle.dump((train_loader, val_loader, test_loader), f)

    train_ids = [i for dat in train_loader for i in dat[-2]]
    val_ids = [i for dat in val_loader for i in dat[-2]]
    test_ids = [i for dat in test_loader for i in dat[-2]]
    model_name = config.base_config.backbone
    dumpjson({"train_id": train_ids, "val_id": val_ids, "test_id": test_ids},
             os.path.join(output_path, f"train_test_ids_{model_name}_{config.dataset}_{config.target}.json"))

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    if classification:
        config.base_config.classification = True

    if model is None:
        net = CrAKN(config.base_config).backbone
    else:
        net = model

    net.to(device)
    optimizer = setup_optimizer(net.parameters(), config)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=0.1)

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

    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
    }
    criterion = criteria[config.criterion]
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    sample_targets = torch.concat([data[-1] for data in tqdm(train_loader, "Normalizing..")], dim=0).squeeze()
    normalizer = Normalizer(sample_targets)

    # train the model!
    for epoch in range(config.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = [AverageMeter() for _ in range(len(config.target))]
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
            output = net(train_inputs, output_level="property")
            loss = torch.zeros(1).to(device)

            for i, prop in enumerate(config.target):
                inds = torch.where(torch.logical_not(torch.isnan(target[:, i])))[0]
                loss += criterion(output[inds, i], target_normed[inds, i])
            #loss = criterion(output, target_normed)
            prediction = normalizer.denorm(output.data.cpu())

            prop_mae_errors, prop_mse_errors, prop_rmse_errors = [], [], []
            for i, prop in enumerate(config.target):
                prop_mae_errors.append(mae(prediction[:, i], target[:, i]))
                prop_mse_errors.append(mse(prediction[:, i], target[:, i]))
                prop_rmse_errors.append(rmse(prediction[:, i], target[:, i]))

            for i, _ in enumerate(config.target):
                inds = torch.where(torch.logical_not(torch.isnan(target[:, i])))[0]
                mae_errors[i].update(prop_mae_errors[i].cpu().item(), target[inds, i].size(0))
            losses.update(loss.cpu().item(), target.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if epoch % 25 == 0 and step % 25 == 0:
                for i, t in enumerate(config.target):
                    print('Target: {target}\t\t'
                          'Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                        epoch, step, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, mae_errors=mae_errors[i], target=t)
                    )
        scheduler.step()

    # Test the model

    net.eval()
    f = open(os.path.join(output_path, "prediction_results_test_set.csv"), "w")
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

            out_data = net(test_bb_data, output_level="property").cpu()
            target = target.cpu().numpy().tolist()

            pred = normalizer.denorm(out_data).data.numpy().tolist()
            targets.append(target)
            predictions.append(pred)
        f.close()

        targets = reduce(lambda x, y: x + y, targets)
        predictions = reduce(lambda x, y: x + y, predictions)
        #targets = np.vstack(targets)
        #predictions = np.vstack(predictions)
        print(np.array(targets).shape)
        print(np.array(predictions).shape)

        def mad(target):
            return np.nanmean(np.abs(target - np.nanmean(target)))

        for i, targ in enumerate(config.target):
            prop_targets = np.array(targets)[:, i]
            prop_preds = np.array(predictions)[:, i]
            inds = np.where(np.logical_not(np.isnan(prop_targets)))[0]

            print(f"{targ} Test MAE:",
                  mean_absolute_error(np.array(targets)[inds, i], np.array(predictions)[inds, i]))
            print(f"{targ} Test MAD:",
                  mad(np.array(targets)[:, i]))
            with open("res.txt", "a") as f:
                f.write(
                    f"Test MAE ({config.base_config.backbone}, {config.dataset}, {targ}, {config.base_config.backbone_only}) :"
                    f" {str(mean_absolute_error(np.array(targets)[inds, i], np.array(predictions)[inds, i]))} \n")


        resultsfile = os.path.join(
            output_path, "prediction_results_test_set.csv"
        )

        model_file = os.path.join(output_path,
                                  f"model_{config.base_config.backbone}_{config.dataset}_{config.target}")

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

    train_vlm(config)
