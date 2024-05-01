import json
import pickle
import pprint
import random
import time
from functools import reduce, partial
from typing import Union, Dict

import numpy as np
import torch
from ignite.metrics import Loss, MeanAbsoluteError
from sklearn.metrics import mean_absolute_error
from torch import nn
from tqdm import tqdm

from crakn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse
import os

from crakn.core.data import PretrainCrAKNDataset, convert_to_pretrain_dataset, prepare_crakn_batch, \
    create_test_dataloader
from crakn.core.model import CrAKN
from crakn.train import setup_optimizer
from crakn.utils import mae, mse, rmse, Normalizer, AverageMeter

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

parser.add_argument(
    "--model-path",
    default="",
    help="Path to saved Torch Model"
)


def train_crakn(model_path: str, config: Union[TrainingConfig, Dict], return_predictions=False):
    import os
    suffix = f"{config.base_config.backbone}_{config.dataset}_{'_'.join(config.target)}"
    model_file = f"model_{suffix}"
    dataloader_file = f"dataloader_{suffix}"
    id_file = f"train_test_ids_{suffix}.json"
    dataloader_path = os.path.join(model_path, dataloader_file)
    train_test_id_path = os.path.join(model_path, id_file)

    vlm = torch.load(os.path.join(model_path, model_file))
    vlm.eval()

    with open(dataloader_path, "rb") as f:
        train_loader, val_loader, test_loader = pickle.load(f)

    #train_loader, val_loader, test_loader = convert_to_pretrain_dataset(vlm, train_loader, val_loader, test_loader,
    #                                                                    config)

    with open(train_test_id_path, "r") as f:
        train_test_ids = json.load(f)

    if type(config) is dict:
        config = TrainingConfig(**config)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    tmp = config.dict()
    f = open(os.path.join(config.output_dir,
                          f"config_crakn_{config.base_config.backbone}_{config.dataset}_{'_'.join(config.target)}.json"),
             "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()

    net = CrAKN(config.base_config)

    net.to(device)
    vlm.to(device)
    net.backbone = vlm
    net.backbone.eval()

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

    sample_targets = torch.concat([data[-1][config.mo_target_index] for data in tqdm(train_loader, "Normalizing..")], dim=0)
    normalizer = Normalizer(sample_targets)

    prepare_batch = partial(prepare_crakn_batch, device=device,
                            internal_prepare_batch=train_loader.dataset.data.prepare_batch,
                            variable=config.variable_batch_size)


    # train the model!
    for epoch in range(config.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = AverageMeter()
        end = time.time()
        net.train()
        for step, dat in enumerate(train_loader):
            if config.base_config.mtype == "Transformer":
                X, target = prepare_batch(dat)
                target = target[:, config.mo_target_index, None]

                samples = random.randint(2, config.batch_size)
                if config.variable_batch_size:
                    nf = nf[:samples]
                    amds = amds[:samples]
                    latt = latt[:samples]
                    ids = ids[:samples]
                    target = target[:samples]

                target_normed = normalizer.norm(target).to(device)
                output = net(X, pretrained=True)
                loss = criterion(output, target_normed)
            else:
                g, original_ids, node_ids, ids, target = dat
                target_normed = normalizer.norm(target).to(device)
                output = net((g.to(device), original_ids.to(device), node_ids.to(device)))
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

            if epoch % 1 == 0 and step % 5 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, step, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
        scheduler.step()

    net.eval()
    f = open(os.path.join(model_path, "crakn_prediction_results_test_set.csv"), "w")
    f.write("id,target,prediction\n")
    targets = []
    predictions = []

    with torch.no_grad():
        if config.base_config.mtype == "Transformer":
            test_data = create_test_dataloader(net, train_loader, test_loader, prepare_batch, config.max_neighbors)

            for dat in tqdm(test_data, desc="Predicting on test set.."):
                X_test, target = prepare_batch(dat)
                target = target[:, config.mo_target_index, None]

                temp_pred = net(X_test)
                prediction = temp_pred[-1:].cpu()
                pred = normalizer.denorm(prediction).data.numpy().flatten().tolist()
                target = target.cpu().numpy().flatten().tolist()[-1:]
                targets.append(target)
                predictions.append(pred)
        else:
            for dat in tqdm(test_loader, desc="Predicting on test set.."):
                g, original_ids, node_ids, ids, target = dat
                out_data = net((g.to(device), original_ids.to(device), node_ids.to(device)))
                target = target.cpu().numpy().flatten().tolist()
                pred = normalizer.denorm(out_data.cpu()).data.numpy().flatten().tolist()
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
                f"Test MAE ({config.base_config.backbone}, {config.dataset}, {config.target[config.mo_target_index]}) - Pretrained :"
                f" {str(mean_absolute_error(np.array(targets), np.array(predictions)))} \n")

        resultsfile = os.path.join(
            model_path, "crakn_prediction_results_test_set.csv"
        )

        model_file = os.path.join(model_path,
                                  f"model_crakn_{config.base_config.backbone}_{config.dataset}_{config.target}")

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


if __name__ == "__main__":
    import sys

    args = parser.parse_args(sys.argv[1:])
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

    print(config.base_config.backbone_config.outputs)
    train_crakn(args.model_path, config)
