import os
import random
import time
from functools import reduce, partial
from typing import Union, Dict, Any, Tuple

import dgl
import numpy as np
import torch
from ignite.metrics import MeanAbsoluteError, Loss
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from crakn.backbones.gnn import GNNData
from crakn.backbones.graphs import dg
from crakn.core.data import CrAKNDataset, get_dataloader

from crakn.config import TrainingConfig
from crakn.core.model import get_backbone
from crakn.train import train_crakn, group_decay, setup_optimizer
from jarvis.db.jsonutils import loadjson
from scipy.io import loadmat
import pickle
import argparse
from scipy.spatial.distance import pdist, squareform

from crakn.utils import mae, mse, rmse, AverageMeter, Normalizer, count_parameters

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
    "--path",
    default="",
    help=".mat filepath"
)

parser.add_argument(
    "--k",
    default=None,
    help="k-Nearest Neighbors"
)

parser.add_argument(
    "--col_tol",
    default=None,
    help="collapse tolerance"
)

parser.add_argument(
    "--ns",
    default=None,
    help="Neighbor Strategy"
)


def cloud_to_graph(x: np.array, k: int) -> dgl.graph:
    dist_mat = squareform(pdist(x))
    k = min(k, dist_mat.shape[0])
    closest = np.argsort(dist_mat, axis=1)[:, :k]
    distances = np.sort(dist_mat, axis=1)[:, :k].reshape(-1)
    neighbors = closest.reshape(-1)
    v = np.array(neighbors)
    u = [i for i in range(dist_mat.shape[0]) for _ in range(k)]
    u = np.array(u)
    g = dgl.graph(data=[], num_nodes=dist_mat.shape[0])
    g.add_edges(u, v)
    g.edata["distance"] = torch.Tensor(distances)
    return g


def load_qm7(path: str, config: TrainingConfig, coord_key="R", target_key="T", atom_type_key="Z"):
    data = loadmat(path)
    coords = data[coord_key]
    targets = data[target_key].reshape(-1)
    atom_types = data[atom_type_key]
    ids = np.array(list(range(targets.shape[0])))

    graphs = []
    num_edges = []
    num_nodes = []
    dg_num_nodes = []
    dg_num_edges = []

    use_dg = config.base_config.backbone_config.neighbor_strategy == "ddg"

    for index in tqdm(range(targets.shape[0]), desc=f"Creating Molecule Graphs: ${use_dg}"):
        padding_start_index = np.where(atom_types[index] == 0)[0]
        if padding_start_index.shape[0] == 0:
            padding_start_index = coords[index].shape[0]
        else:
            padding_start_index = np.min(padding_start_index)

        unpadded_coords = coords[index][:padding_start_index]
        unpadded_atom_types = atom_types[index][:padding_start_index]

        g = cloud_to_graph(unpadded_coords, config.base_config.backbone_config.max_neighbors)
        g.ndata["atom_features"] = torch.Tensor(unpadded_atom_types)

        num_nodes.append(g.num_nodes())
        num_edges.append(g.num_edges())
        if use_dg:
            g = dg(g, collapse_tol=config.base_config.backbone_config.collapse_tol)
            dg_num_nodes.append(g.num_nodes())
            dg_num_edges.append(g.num_edges())
        else:
            g.ndata["weights"] = torch.ones(g.num_nodes())

        graphs.append(g)

    dg_num_nodes = np.array(dg_num_nodes)
    dg_num_edges = np.array(dg_num_edges)
    num_edges = np.array(num_edges)
    num_nodes = np.array(num_nodes)
    if config.base_config.backbone_config.neighbor_strategy == "ddg":
        print(f"Reduction in vertices: {np.nanmean(dg_num_nodes / num_nodes)}")
        print(f"Reduction in edges: {np.nanmean(dg_num_edges / num_edges)}")

    return GNNData([], targets, ids, graphs=graphs)


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

    collate_fn = dataset.collate_fn

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

    test_set = torch.utils.data.Subset(dataset, indices[-test_size:])
    test_loader = DataLoader(test_set, batch_size=config.test_batch_size,
                             num_workers=config.num_workers,
                             collate_fn=collate_fn, pin_memory=config.pin_memory)
    return train_loader, val_loader, test_loader


def train_qm7(config: TrainingConfig, path: str):
    dataset = load_qm7(path, config)
    train_loader, val_loader, test_loader = get_dataloader(dataset, config)
    train_(config, (train_loader, val_loader, test_loader))


def train_(
        config: Union[TrainingConfig, Dict[str, Any]],
        dataloaders: Tuple[DataLoader, DataLoader, DataLoader],
        model: nn.Module = None,
        return_predictions: bool = False):

    import os

    if type(config) is dict:
        config = TrainingConfig(**config)


    tmp = config.dict()


    if config.random_seed is not None:
        deterministic = True
        torch.cuda.manual_seed_all(config.random_seed)

    train_loader, val_loader, test_loader = dataloaders

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    prepare_batch = partial(train_loader.dataset.prepare_batch, device=device)

    if model is None:
        net = get_backbone(
            config.base_config.backbone,
            bb_config=config.base_config.backbone_config
        )
    else:
        net = model

    #num_parameters = count_parameters(net)

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
        mae_errors = AverageMeter()
        end = time.time()
        net.train()
        for step, dat in enumerate(train_loader):
            X, target = prepare_batch(dat)
            target_normed = normalizer.norm(target).to(device)
            output = net(X, target_normed)

            loss = criterion(torch.squeeze(output), torch.squeeze(target_normed))

            prediction = normalizer.denorm(output.data.cpu())
            target = target.cpu()

            prop_mae_errors, prop_mse_errors, prop_rmse_errors = [], [], []
            for i, prop in enumerate(config.target):
                prop_mae_errors.append(mae(prediction[i], target[i]))
                prop_mse_errors.append(mse(prediction[i], target[i]))
                prop_rmse_errors.append(rmse(prediction[i], target[ i]))

            mae_errors.update(prop_mae_errors[i].cpu().item(), target.size(0))

            losses.update(loss.cpu().item(), target.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if epoch % 100 == 0 and step % 25 == 0:
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

    targets = []
    predictions = []

    with torch.no_grad():

        for dat in tqdm(test_loader, desc="Predicting on test set.."):
            X, target = prepare_batch(dat)
            output = net(X, "")
            target = target.cpu().numpy().flatten().tolist()
            pred = normalizer.denorm(output.cpu()).data.numpy().flatten().tolist()
            targets.append(target)
            predictions.append(pred)

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
                f" {str(mean_absolute_error(np.array(targets), np.array(predictions)))} \n")

        target_vals = np.array(targets, dtype="float").flatten()
        predictions = np.array(predictions, dtype="float").flatten()

    if return_predictions:
        return history, predictions
    return history


if __name__ == '__main__':
    import sys

    args = parser.parse_args(sys.argv[1:])

    if args.config != "":
        config = loadjson(args.config)
    else:
        config = TrainingConfig()

    if args.k is not None:
        print(args.k)
        config["base_config"]["backbone_config"]["max_neighbors"] = int(args.k)

    if args.col_tol is not None:
        print(args.col_tol)
        config["base_config"]["backbone_config"]["collapse_tol"] = float(args.col_tol)

    if args.ns is not None:
        print(args.ns)
        config["base_config"]["backbone_config"]["neighbor_strategy"] = args.ns

    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    train_qm7(config, args.path)
