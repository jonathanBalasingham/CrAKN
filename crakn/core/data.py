from crakn.core.config import TrainingConfig
from typing import List
from pymatgen.core.structure import Structure
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
# from matbench.bench import MatbenchBenchmark
import torch
import dgl


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


class StructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(self, kg: dgl.DGLGraph, id_tag="jid"):
        """Pytorch Dataset for CrAKN.

        """

    def __len__(self):
        """Get length."""
        return self.kg.num_nodes()

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        return self.kg.sub_graph(idx)

    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
            samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.tensor(labels)


if __name__ == "__main__":
    fake_config = TrainingConfig()
    structures, targets = retrieve_data(fake_config)
