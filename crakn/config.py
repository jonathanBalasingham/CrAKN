"""Pydantic model for default configuration and validation."""

import subprocess
from typing import Optional, Union, List
import os
from pydantic import model_validator
from typing import Literal

from crakn.backbones.pst import PSTConfig
from crakn.utils import BaseSettings
from crakn.backbones.gcn import SimpleGCNConfig
from crakn.core.model import CrAKNConfig

try:
    VERSION = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )
except Exception:
    VERSION = "NA"
    pass


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92, "mat2vec": 200}


TARGET_ENUM = Literal[
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
    "gap pbe",
    "e_form",
    "e_hull",
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
    "e_above_hull",
    "mu_b",
    "bulk modulus",
    "shear modulus",
    "elastic anisotropy",
    "U0",
    "HOMO",
    "LUMO",
    "R2",
    "ZPVE",
    "omega1",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U",
    "H",
    "G",
    "Cv",
    "A",
    "B",
    "C",
    "all",
    "target",
    "max_efg",
    "avg_elec_mass",
    "avg_hole_mass",
    "_oqmd_band_gap",
    "_oqmd_delta_e",
    "_oqmd_stability",
    "edos_up",
    "pdos_elast",
    "bandgap",
    "energy_total",
    "net_magmom",
    "b3lyp_homo",
    "b3lyp_lumo",
    "b3lyp_gap",
    "b3lyp_scharber_pce",
    "b3lyp_scharber_voc",
    "b3lyp_scharber_jsc",
    "log_kd_ki",
    "max_co2_adsp",
    "min_co2_adsp",
    "lcd",
    "pld",
    "void_fraction",
    "surface_area_m2g",
    "surface_area_m2cm3",
    "indir_gap",
    "f_enp",
    "final_energy",
    "energy_per_atom",
    "ead",
]


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = VERSION

    # dataset configuration
    dataset: Literal[
        "dft_3d",
        "dft_3d_2021",
        "dft_2d",
        "matbench"
    ] = "dft_3d_2021"
    target: List[TARGET_ENUM] = ["exfoliation_energy"]
    pretrain_target: List[TARGET_ENUM] = ["formation_energy_peratom"]
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn", "mat2vec"] = "mat2vec"
    neighbor_strategy: Literal[
        "k-nearest", "ddg", "mdg"
    ] = "k-nearest"
    id_tag: Literal["jid", "id", "_oqmd_entry_id"] = "jid"
    prediction_method: Literal["single", "ensemble"] = "ensemble"
    n_ensemble: int = 10

    # logging configuration

    # training configuration
    random_seed: Optional[int] = 123
    classification_threshold: Optional[float] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    target_multiplication_factor: Optional[float] = None
    epochs: int = 250
    batch_size: int = 32
    variable_batch_size: bool = False
    test_batch_size: int = 3
    weight_decay: float = 1e-5
    learning_rate: float = 1e-4
    filename: str = "sample"
    warmup_steps: int = 2000
    criterion: Literal["mse", "l1", "poisson", "zig"] = "l1"
    optimizer: Literal["adamw", "sgd", "adam"] = "adamw"
    scheduler: Literal["onecycle", "step", "none"] = "step"
    pin_memory: bool = False
    save_dataloader: bool = False
    write_checkpoint: bool = True
    write_predictions: bool = True
    store_outputs: bool = True
    progress: bool = True
    log_tensorboard: bool = False
    standard_scalar_and_pca: bool = False
    use_canonize: bool = True
    num_workers: int = 4
    cutoff: float = 8.0
    cutoff_extra: float = 3.0
    max_neighbors: int = 12
    keep_data_order: bool = False
    normalize_graph_level_loss: bool = False
    distributed: bool = False
    data_parallel: bool = False
    n_early_stopping: Optional[int] = None  # typically 50
    output_dir: str = os.path.abspath("../temp")
    lr_milestones: List[int] = [150, 200, 400]
    mo_target_index: int = 0
    extra_features: List[TARGET_ENUM] = ["formation_energy_peratom", "exfoliation_energy"]
    composition_features: Literal["mat2vec", "cgcnn"] = "cgcnn"
    # model configuration
    base_config: CrAKNConfig = CrAKNConfig(name="crakn")

    @model_validator(mode="after")
    @classmethod
    def set_input_size(cls, values):
        values.base_config.backbone_config.atom_input_features = FEATURESET_SIZE[
            values.atom_features
        ]
        values.base_config.backbone_config.outputs = len(values.target)
        values.base_config.extra_features = len(values.extra_features)
        if values.atom_features == "mat2vec":
            values.composition_features = "cgcnn"
        else:
            values.composition_features = "mat2vec"

        values.base_config.comp_feat_size = FEATURESET_SIZE[values.composition_features]
        assert values.batch_size > 1

        return values

