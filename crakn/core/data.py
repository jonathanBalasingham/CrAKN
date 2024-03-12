from crakn.core.config import TrainingConfig
from typing import List
from pymatgen.core.structure import Structure
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
#from matbench.bench import MatbenchBenchmark


def retrieve_data(config: TrainingConfig) -> tuple[List[Structure], List[float]]:
    if config.dataset == "matbench":
        pass
    else:
        d = data(config.dataset)

    structures: List[Structure] = []
    targets: List[float] = []
    for datum in d:
        if config.target not in datum.keys():
            raise ValueError(f"Unknown target {config.target}")

        target = datum[config.target]
        atoms = (Atoms.from_dict(datum["atoms"]) if isinstance(datum["atoms"], dict) else datum["atoms"])
        structure = atoms.pymatgen_converter()
        structures.append(structure)
        targets.append(target)
    return structures, targets


if __name__ == "__main__":
    fake_config = TrainingConfig()
    structures, targets = retrieve_data(fake_config)
