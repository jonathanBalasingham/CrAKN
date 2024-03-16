import dgl
import amd
import torch
from tqdm import tqdm
import numpy as np
from crakn.backbones import *


def knowledge_graph(structures, targets,
                    add_edges: bool = False,
                    nearest_neighbors: int = 5,
                    cutoff: float = 0.01,
                    metric_knn: int = 100,
                    include_lattice: bool = True,
                    crystal_representation: str = "PST") -> dgl.graph:
    """
    builds the Knowledge graph:

    Parameters:
        structures - list of pymatgen structures
        targets - list of targets
        add_edges - whether to pre-add edges to the graph
        nearest_neighbors - number of nearest neighbors to add directed edges to
        cutoff - cutoff for distances between neighbors, results in a maximum of "nearest neighbors" edges

    Returns:
        graph - dgl.graph with nodes representing crystals and edges representing invariant distances
    """

    num_nodes = len(structures)
    g = dgl.graph(data=[])
    g.add_nodes(num_nodes)

    periodic_sets = [amd.periodicset_from_pymatgen_structure(s) for s in tqdm(structures, desc="Creating Periodic Sets..")]
    amds = np.vstack([amd.AMD(ps, k=metric_knn) for ps in tqdm(periodic_sets, desc="Calculating AMDs..")])
    lattices = [list(s.lattice.parameters) for s in tqdm(structures, desc="Retrieving lattices..")]

    g.ndata['lattice'] = torch.Tensor(lattices)
    g.ndata["amd"] = torch.Tensor(amds)

    if crystal_representation == "PST":
        pdds = [amd.PDD(ps, k=metric_knn) for ps in tqdm(periodic_sets, "Calculating PDDs..")]
        compositions = [np.array(ps.types)[:, None] for ps in tqdm(periodic_sets, "Retrieving compositions..")]
        g.ndata["composition"] = torch.nested.nested_tensor(compositions)
        g.ndata["pdd"] = torch.nested.nested_tensor(pdds)
    else:
        raise NotImplementedError(f"Crystal representation: {crystal_representation}, not implemented")

    g.ndata["target"] = torch.Tensor(targets)
    if not add_edges:
        return g
    else:
        raise NotImplementedError("Pre-added edges not implemented")
