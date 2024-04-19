"""Shared pydantic settings configuration."""
import json
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np

from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic import ConfigDict
import torch
from sklearn import metrics


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""
    model_config = ConfigDict(extra="forbid", use_enum_values=True, env_prefix="jv_")


def plot_learning_curve(
        results_dir: Union[str, Path], key: str = "mae", plot_train: bool = False
):
    """Plot learning curves based on json history files."""
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    with open(results_dir / "history_val.json", "r") as f:
        val = json.load(f)

    p = plt.plot(val[key], label=results_dir.name)

    if plot_train:
        # plot the training trace in the same color, lower opacity
        with open(results_dir / "history_train.json", "r") as f:
            train = json.load(f)

        c = p[0].get_color()
        plt.plot(train[key], alpha=0.5, c=c)

    plt.xlabel("epochs")
    plt.ylabel(key)

    return train, val


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


class Normalizer(object):

    def __init__(self, tensor):
        self.mean = torch.nanmean(tensor, dim=0)
        self.std = nanstd(tensor, dim=0)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class SigmoidNormalizer(object):

    def __init__(self, tensor) -> None:
        self.min = torch.min(tensor)
        self.max = torch.max(tensor)

    def norm(self, tensor) -> torch.Tensor:
        return (tensor - self.min) / (self.max - self.min)

    def denorm(self, normed_tensor) -> torch.Tensor:
        return ((self.max - self.min) * normed_tensor) + self.min

    def state_dict(self):
        return {'min': self.min,
                'max': self.max}

    def load_state_dict(self, state_dict):
        self.min = state_dict['min']
        self.max = state_dict['max']


def mae(prediction, target):
    return torch.nanmean(torch.abs(target - prediction))


def mse(prediction, target):
    return torch.nanmean((target - prediction) ** 2)


def rmse(prediction, target):
    return torch.sqrt(mse(prediction, target))


def mape(prediction, target):
    return torch.nanmean(torch.abs((target - prediction) / target))


def mad(target):
    return torch.nanmean(torch.abs(target - torch.mean(target)))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
