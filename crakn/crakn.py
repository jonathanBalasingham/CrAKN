import amd

from crakn.utils import BaseSettings
from typing import Literal, Optional
from pathlib import Path
import dgl
import tqdm
from jarvis.core.atoms import Atoms


class CrAKNConfig(BaseSettings):
    name: Literal["crakn"]


def create_representation(structure, model):
    pass


def knowledge_graph(
        dataset=[],
        name: str = "dft_3d",
        neighbor_strategy: str = "k-nearest",
        cutoff: float = 8,
        max_neighbors: int = 12,
        cachedir: Optional[Path] = None,
        #config: TrainingConfig = None,
):

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{neighbor_strategy}-kg.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        # print('dataset',dataset,type(dataset))
        print("Creating Knowledge Graph")
        graphs = []
        representations = []
        periodic_sets = []
        amds = []
        for ii, i in tqdm(dataset.iterrows()):
            atoms = i["atoms"]
            structure = (
                Atoms.from_dict(atoms) if isinstance(atoms, dict) else atoms
            )
            pmg = i["atoms"].pymatgen_converter()
            ps = amd.periodicset_from_pymatgen_structure(pmg)
            periodic_sets.append(ps)
            amds.append(amd.AMD(ps, k=max_neighbors))
            representations.append(create_representation(structure, ))

        # df = pd.DataFrame(dataset)
        # print ('df',df)

        # graphs = df["atoms"].progress_apply(atoms_to_graph).values
        # print ('graphs',graphs,graphs[0])
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs