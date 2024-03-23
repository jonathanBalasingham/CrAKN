from functools import partial, reduce

from typing import Any, Dict, Union, Tuple
import ignite
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from crakn.core.model import CrAKN

try:
    from ignite.contrib.handlers.stores import EpochOutputStore
except Exception:
    from ignite.handlers.stores import EpochOutputStore

    pass

from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn

from crakn.core.data import get_dataloader, retrieve_data, CrAKNDataset, prepare_crakn_batch
from crakn.config import TrainingConfig

from jarvis.db.jsonutils import dumpjson
import json
import pprint

import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

torch.set_default_dtype(torch.float32)


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
        ignite.utils.manual_seed(config.random_seed)

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
        net = CrAKN(config.base_config)
    else:
        net = model

    net.to(device)
    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, )

    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
    }
    criterion = criteria[config.criterion]

    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}
    if config.base_config.output_features > 1 and config.standard_scalar_and_pca:
        metrics = {
            "loss": Loss(
                criterion, output_transform=make_standard_scalar_and_pca
            ),
            "mae": MeanAbsoluteError(
                output_transform=make_standard_scalar_and_pca
            ),
        }

    if config.criterion == "zig":
        def zig_prediction_transform(x):
            output, y = x
            return criterion.predict(output), y

        metrics = {
            "loss": Loss(criterion),
            "mae": MeanAbsoluteError(
                output_transform=zig_prediction_transform
            ),
        }

    if classification:
        criterion = nn.NLLLoss()

        metrics = {
            "accuracy": Accuracy(
                output_transform=thresholded_output_transform
            ),
            "precision": Precision(
                output_transform=thresholded_output_transform
            ),
            "recall": Recall(output_transform=thresholded_output_transform),
            "rocauc": ROC_AUC(output_transform=activated_output_transform),
            "roccurve": RocCurve(output_transform=activated_output_transform),
            "confmat": ConfusionMatrix(
                output_transform=thresholded_output_transform, num_classes=2
            ),
        }
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
    )

    evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch_test,
        device=device,
    )

    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
    )

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        if classification:
            def cp_score(engine):
                """Higher accuracy is better."""
                return engine.state.metrics["accuracy"]

        else:
            def cp_score(engine):
                """Lower MAE is better."""
                return -engine.state.metrics["mae"]

        # save last two epochs
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            Checkpoint(
                to_save,
                DiskSaver(
                    checkpoint_dir, create_dir=True, require_empty=False
                ),
                n_saved=2,
                global_step_transform=lambda *_: trainer.state.epoch,
            ),
        )
        # save best model
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            Checkpoint(
                to_save,
                DiskSaver(
                    checkpoint_dir, create_dir=True, require_empty=False
                ),
                filename_pattern="best_model.{ext}",
                n_saved=1,
                global_step_transform=lambda *_: trainer.state.epoch,
                score_function=cp_score,
            ),
        )
    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    if config.store_outputs:
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        train_evaluator.run(train_loader)
        evaluator.run(val_loader)

        tmetrics = train_evaluator.state.metrics
        vmetrics = evaluator.state.metrics
        for metric in metrics.keys():
            tm = tmetrics[metric]
            vm = vmetrics[metric]
            if metric == "roccurve":
                tm = [k.tolist() for k in tm]
                vm = [k.tolist() for k in vm]
            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()
                vm = vm.cpu().numpy().tolist()

            history["train"][metric].append(tm)
            history["validation"][metric].append(vm)

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history["train"],
            )
        if config.progress:
            pbar = ProgressBar()
            if not classification:
                pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
                pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
            else:
                pbar.log_message(f"Train ROC AUC: {tmetrics['rocauc']:.4f}")
                pbar.log_message(f"Val ROC AUC: {vmetrics['rocauc']:.4f}")

    if config.n_early_stopping is not None:
        if classification:
            def es_score(engine):
                """Higher accuracy is better."""
                return engine.state.metrics["accuracy"]
        else:
            def es_score(engine):
                """Lower MAE is better."""
                return -engine.state.metrics["mae"]

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=es_score,
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)

    # train the model!
    trainer.run(train_loader, max_epochs=config.epochs)

    if config.write_predictions and classification:
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                # out_data = torch.exp(out_data.cpu())
                top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                target = int(target.cpu().numpy().flatten().tolist()[0])

                f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                targets.append(target)
                predictions.append(
                    top_class.cpu().numpy().flatten().tolist()[0]
                )
        f.close()
        from sklearn.metrics import roc_auc_score

        print("predictions", predictions)
        print("targets", targets)
        print(
            "Test ROCAUC:",
            roc_auc_score(np.array(targets), np.array(predictions)),
        )

    if (config.write_predictions
            and not classification
            and config.base_config.output_features > 1):
        net.eval()
        mem = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(open("sc.pkl", "rb"))
                    out_data = list(
                        sc.transform(np.array(out_data).reshape(1, -1))[0]
                    )  # [0][0]
                target = target.cpu().numpy().flatten().tolist()
                info = {}
                info["id"] = id
                info["target"] = target
                info["predictions"] = out_data
                mem.append(info)
        dumpjson(filename=os.path.join(config.output_dir, "multi_out_predictions.json"),
                 data=mem)

    if (config.write_predictions
            and not classification
            and config.base_config.output_features == 1):
        net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []

        with torch.no_grad():
            neighbor_data = []
            for dat in tqdm(train_loader, desc="Generating knowledge network node features.."):
                bb_data, amds, latt, ids, target = dat
                neighbor_node_features = net.backbone((bbd.to(device) for bbd in bb_data[0]))
                neighbor_data.append((neighbor_node_features, amds, latt, target))

            for dat in tqdm(test_loader, desc="Predicting on test set.."):
                bb_data, amds, latt, ids, target = dat

                if config.prediction_method == "ensemble":
                    out_data = []
                    for neighbor_datum in neighbor_data:
                        temp_pred = net(
                            ([
                                 bb_data[0][0].to(device),
                                 bb_data[0][1].to(device),
                                 torch.zeros(bb_data[1].shape).to(device),
                             ],
                             amds.to(device),
                             latt.to(device),
                             ids),
                            neighbors=(datum.to(device) for datum in neighbor_datum)
                        )
                        out_data.append(temp_pred[-len(ids):])
                    ensemble_predictions = torch.stack(out_data)
                    # print(f"SD of preds: {torch.mean(torch.std(ensemble_predictions, dim=0))}")
                    out_data = torch.mean(ensemble_predictions, dim=0)
                else:
                    out_data = net(
                        ([
                             bb_data[0][0].to(device),
                             bb_data[0][1].to(device),
                             bb_data[1].to(device),
                         ],
                         amds.to(device),
                         latt.to(device),
                         ids),
                    )

                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(
                        open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                    )
                    out_data = sc.transform(np.array(out_data).reshape(-1, 1))[
                        0
                    ][0]
                target = target.cpu().numpy().flatten().tolist()
                if len(target) == 1:
                    target = target[0]

                if isinstance(targets, list):
                    target = target if isinstance(target, list) else [target]
                    targets += target
                    if isinstance(out_data[0], list):
                        out_data = [od[0] for od in out_data]

                    predictions += out_data
                    for id, t, p in zip(ids, target, out_data):
                        f.write("%s, %6f, %6f\n" % (id, t, p))
                else:
                    f.write("%s, %6f, %6f\n" % (id, target, out_data))
                    targets.append(target)
                    predictions.append(out_data)
        f.close()

        print("Test MAE:",
              mean_absolute_error(np.array(targets), np.array(predictions)))

        def mad(target):
            return torch.mean(torch.abs(target - torch.mean(target)))

        print(f"Test MAD: {mad(torch.Tensor(targets))}")

        with open("res.txt", "a") as f:
            f.write(f"Test MAE: {str(mean_absolute_error(np.array(targets), np.array(predictions)))} \n")

        if config.store_outputs and not classification:
            resultsfile = os.path.join(
                config.output_dir, "prediction_results_train_set.csv"
            )

            target_vals, predictions = [], []

            for tgt, pred in history["trainEOS"]:
                target_vals.append(tgt.cpu().numpy().flatten().tolist())
                predictions.append(pred.cpu().numpy().flatten().tolist())

            target_vals = reduce(lambda x, y: x + y, target_vals)
            predictions = reduce(lambda x, y: x + y, predictions)
            target_vals = np.array(target_vals, dtype="float").flatten()
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
