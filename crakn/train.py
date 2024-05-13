import time
from functools import partial, reduce

from typing import Any, Dict, Union, Tuple
import ignite
import torch
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from crakn.core.model import CrAKN
from crakn.utils import Normalizer, AverageMeter, mae, mse, rmse, count_parameters

try:
    from ignite.contrib.handlers.stores import EpochOutputStore
except Exception:
    from ignite.handlers.stores import EpochOutputStore
    pass

import pickle as pk
import numpy as np
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from prettytable import PrettyTable

from crakn.core.data import get_dataloader, retrieve_data, CrAKNDataset, prepare_crakn_batch, create_test_dataloader
from crakn.config import TrainingConfig

from jarvis.db.jsonutils import dumpjson
import json
import pprint

import os
import warnings
import pickle

warnings.filterwarnings("ignore", category=RuntimeWarning)

torch.set_default_dtype(torch.float32)



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


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(
        sc.transform(y_pred.cpu().numpy()), device=y_pred.device
    )
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=y.device)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def train_crakn(
        config: Union[TrainingConfig, Dict[str, Any]],
        model: nn.Module = None,
        dataloaders: Tuple[DataLoader, DataLoader, DataLoader] = None,
        return_predictions: bool = False):

    import os

    if type(config) is dict:
        config = TrainingConfig(**config)

    output_directory = f"{config.base_config.backbone}_{config.dataset}_{'_'.join(config.target)}"
    output_path = os.path.join(config.output_dir, output_directory)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

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

    prepare_batch = partial(prepare_crakn_batch, device=device,
                            internal_prepare_batch=train_loader.dataset.data.prepare_batch,
                            variable=config.variable_batch_size)

    prepare_batch_test = partial(prepare_crakn_batch, device=device,
                                 internal_prepare_batch=train_loader.dataset.data.prepare_batch,
                                 variable=False)
    if classification:
        config.base_config.classification = True

    if model is None:
        net = CrAKN(config.base_config)
    else:
        net = model

    if config.base_config.backbone_only:
        num_parameters = count_parameters(net.backbone)
    else:
        num_parameters = count_parameters(net)

    net.to(device)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

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
    sample_targets = torch.concat([data[-1] for data in tqdm(train_loader, "Normalizing..")], dim=0).squeeze()
    normalizer = Normalizer(sample_targets)

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    # train the model!
    for epoch in range(config.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mae_errors = [AverageMeter() for _ in range(len(config.target))]
        end = time.time()
        net.train()
        for step, dat in enumerate(train_loader):
            X, target = prepare_batch(dat)
            target_normed = normalizer.norm(target).to(device)
            output = net(X, target_normed)
            loss = torch.zeros(1).to(device)

            for i, prop in enumerate(config.target):
                inds = torch.where(torch.logical_not(torch.isnan(target[:, i])))[0]
                loss += criterion(output[inds, i], target_normed[inds, i])

            prediction = normalizer.denorm(output.data.cpu())
            target = target.cpu()

            prop_mae_errors, prop_mse_errors, prop_rmse_errors = [], [], []
            for i, prop in enumerate(config.target):
                prop_mae_errors.append(mae(prediction[:, i], target[:, i]))
                prop_mse_errors.append(mse(prediction[:, i], target[:, i]))
                prop_rmse_errors.append(rmse(prediction[:, i], target[:, i]))

            for i, _ in enumerate(config.target):
                inds = torch.where(torch.logical_not(torch.isnan(target[:, i])))[0]
                error_update_size = target[inds, i].size(0)
                if error_update_size > 0:
                    mae_errors[i].update(prop_mae_errors[i].cpu().item(), error_update_size)
            losses.update(loss.cpu().item(), target.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if epoch % 5 == 0 and step % 25 == 0:
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

    for i in [4, 8, 16, 32, 64, 128]:
            net.eval()
            f = open(os.path.join(output_path, "crakn_prediction_results_test_set.csv"), "w")
            f.write("id,target,prediction\n")
            targets = []
            predictions = []

            with torch.no_grad():

                if config.base_config.mtype == "Transformer":
                    test_data = create_test_dataloader(net, train_loader, test_loader, prepare_batch, i)

                    for dat in tqdm(test_data, desc="Predicting on test set.."):
                        X_test, target = prepare_batch(dat)

                        temp_pred = net(X_test, normalizer.norm(target))
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

                targets = reduce(lambda x, y: x + y, [i[-1].cpu().numpy().flatten().tolist() for i in test_loader])
                predictions = reduce(lambda x, y: x + y, predictions)
                print(f"Number of Predictions: {len(targets)}, {len(test_loader.dataset)}")
                print("Test MAE:",
                      mean_absolute_error(np.array(targets), np.array(predictions)))

                def mad(target):
                    return torch.mean(torch.abs(target - torch.mean(target)))

                print(f"Test MAD: {mad(torch.Tensor(targets))}")

                with open("res.txt", "a") as f:
                    f.write(
                        f"Test MAE ({config.base_config.backbone}, {config.dataset}, {config.target}, {config.base_config.backbone_only}) :"
                        f" {str(mean_absolute_error(np.array(targets), np.array(predictions)))} ({num_parameters}) \n")

                resultsfile = os.path.join(
                    config.output_dir, "crakn_prediction_results_test_set.csv"
                )

                model_file = os.path.join(output_path,
                                          f"full_model_crakn_{config.base_config.backbone}_{config.dataset}_{config.target}")

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
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_crakn(config)
