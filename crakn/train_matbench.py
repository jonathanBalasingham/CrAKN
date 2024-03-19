import torch
from crakn.config import TrainingConfig
from crakn.core.data import CrAKNDataset, get_dataloader
from crakn.train import train_crakn
from jarvis.db.jsonutils import loadjson
from matbench.bench import MatbenchBenchmark
import argparse
import time
import numpy as np
import pickle
import pandas as pd


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


def run_fold(fold, task, config: TrainingConfig, suffix: str = ""):
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
    test_size = test_inputs.shape[0]
    val_size = int(len(train_outputs) * config.val_ratio)
    train_size = len(train_outputs) - val_size

    config_as_dict = config.dict()
    config_as_dict["train_size"] = train_size
    config_as_dict["val_size"] = val_size
    config_as_dict["test_size"] = test_size

    structures = pd.concat([train_inputs, test_inputs])
    targets = pd.concat([train_outputs, test_outputs * 0])
    ids = list(range(targets.shape[0]))
    crakn_dataset = CrAKNDataset(structures, targets, ids, config)

    start_time = time.time()
    history, predictions = train_crakn(config, dataloaders=get_dataloader(crakn_dataset, config))
    end_time = time.time()
    print(f"Fold took {(end_time - start_time) / 60} minutes")
    print('---------Evaluate Model on Test Set---------------')
    #start_time_pred = time.time()

    #end_time_pred = time.time()
    #print(f"Prediction time: {end_time_pred - start_time_pred} seconds")

    with open(f"{task.dataset_name}_fold{fold}_predictions_compute_times_{suffix}.txt", "w") as f:
        f.write(f"Training time: {(end_time - start_time) / 60} minutes\n")
        #f.write(f"Prediction time: {end_time_pred - start_time_pred} seconds\n")

    return predictions


def train_matbench(config: TrainingConfig, task, verbose: bool = True, suffix: str = ""):
    task.load()
    fold_times = []
    for fold in task.folds:
        st = time.time()
        predictions = run_fold(fold, task, config, suffix=suffix)
        end = time.time()
        if verbose:
            print(f"Fold took {end - st} seconds")
            fold_times.append(end - st)

        with open(f"{task.dataset_name}_fold{fold}_predictions_{suffix}", "wb") as f:
            pickle.dump(predictions, f)

        task.record(fold, predictions)

    if verbose:
        print(task.scores)

    with open(f"{task.dataset_name}_mat2vec_v2_results_{suffix}.txt", "w") as f:
        f.write(str(task.scores))
        f.write("\n")
        f.write(str(fold_times))
        f.write("\n")
        f.write(str(np.mean(fold_times)))

    my_metadata = {
        "CrAKN": "v0.1",
        "configuration": config.dict()
    }
    mb.add_metadata(my_metadata)
    tasks = [task]
    mb.to_file(f"results_v2_{'_'.join(t.dataset_name for t in tasks)}_{suffix}.json.gz")
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

    mb = MatbenchBenchmark(autoload=False)

    tasks = {"matbench_phonons": mb.matbench_phonons,
             "matbench_jdft2d": mb.matbench_jdft2d,
             "matbench_dielectric": mb.matbench_dielectric,
             "matbench_log_gvrh": mb.matbench_log_gvrh,
             "matbench_log_kvrh": mb.matbench_log_kvrh,
             "matbench_perovskites": mb.matbench_perovskites,
             "matbench_mp_e_form": mb.matbench_mp_e_form,
             "matbench_mp_gap": mb.matbench_mp_gap
             }

    if config.target not in tasks:
        raise ValueError(f"Target {config.target} is not in Matbench")

    task = tasks[config.target]
    train_matbench(config, task)
