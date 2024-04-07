from multiprocessing.context import ForkContext
from pathlib import Path
from re import X
import numpy as np
import pandas as pd
import torch_geometric
from jarvis.core.specie import chem_data, get_node_attributes

# from jarvis.core.atoms import Atoms
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional, Literal
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.data.batch import Batch
import itertools

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter

from crakn.backbones.utils import RBFExpansion, angle_emb_mp
from crakn.utils import BaseSettings
import pandas as pd
import dgl

try:
    import torch
    from tqdm import tqdm
except Exception as exp:
    print("torch/tqdm is not installed.", exp)
    pass


class MatformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["Matformer"]
    conv_layers: int = 5
    edge_layers: int = 0
    atom_input_features: int = 92
    edge_features: int = 128
    triplet_input_features: int = 40
    node_features: int = 128
    fc_layers: int = 1
    fc_features: int = 128
    output_features: int = 1
    node_layer_head: int = 4
    edge_layer_head: int = 4
    nn_based: bool = False

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False
    atom_features: str = "cgcnn"
    transform: str = ""
    line_graph: bool = True
    neighbor_strategy: str = "k-nearest"
    lineControl: bool = True
    max_neighbors: int = 12
    cutoff: float = 8
    use_canonize: bool = False
    use_lattice: bool = False
    compute_line_graph: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class Matformer(nn.Module):
    """att pyg implementation."""

    def __init__(self, config: MatformerConfig = MatformerConfig(name="Matformer")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        print(config)
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features),
        )
        self.angle_lattice = config.angle_lattice
        if self.angle_lattice:  ## module not used
            print('use angle lattice')
            self.lattice_rbf = nn.Sequential(
                RBFExpansion(
                    vmin=0,
                    vmax=8.0,
                    bins=config.edge_features,
                ),
                nn.Linear(config.edge_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_angle = nn.Sequential(
                RBFExpansion(
                    vmin=-1,
                    vmax=1.0,
                    bins=config.triplet_input_features,
                ),
                nn.Linear(config.triplet_input_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_emb = nn.Sequential(
                nn.Linear(config.node_features * 6, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_atom_emb = nn.Sequential(
                nn.Linear(config.node_features * 2, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )


        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=config.node_features, out_channels=config.node_features,
                              heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.edge_update_layers = nn.ModuleList(  ## module not used
            [
                MatformerConv(in_channels=config.node_features, out_channels=config.node_features,
                              heads=config.edge_layer_head, edge_dim=config.node_features)
                for _ in range(config.edge_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                config.fc_features, config.output_features
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, data) -> torch.Tensor:
        data, ldata, lattice = data
        # initial node features: atom feature network...

        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)

        edge_features = self.rbf(edge_feat)
        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        #node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        #node_features = self.att_layers[4](node_features, data.edge_index, edge_features)

        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        # features = F.softplus(features)
        features = self.fc(features)

        out = self.fc_out(features)
        if self.link:
            out = self.link(out)
        if self.classification:
            out = self.softmax(out)

        return features


class MatformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.0,
            edge_dim: Optional[int] = None,
            bias: bool = True,
            root_weight: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(MatformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], out_channels,
                                      bias=bias)
            self.lin_concate = nn.Linear(heads * out_channels, out_channels)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(nn.Linear(out_channels * 3, out_channels), nn.LayerNorm(out_channels))
        # self.msg_layer = nn.Linear(out_channels * 3, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        # self.bn = nn.BatchNorm1d(out_channels * heads)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(out_channels * 3)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.concat:
            self.lin_concate.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.concat:
            out = self.lin_concate(out)

        out = F.silu(self.bn(out))  # after norm and silu

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_i: Tensor, key_j: Tensor, value_j: Tensor, value_i: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
        query_i = torch.cat((query_i, query_i, query_i), dim=-1)
        key_j = torch.cat((key_i, key_j, edge_attr), dim=-1)
        alpha = (query_i * key_j) / math.sqrt(self.out_channels * 3)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.cat((value_i, value_j, edge_attr), dim=-1)
        out = self.lin_msg_update(out) * self.sigmoid(
            self.layer_norm(alpha.view(-1, self.heads, 3 * self.out_channels)))
        out = self.msg_layer(out)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class MatformerData(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
            self,
            structures,
            targets,
            config: MatformerConfig = MatformerConfig(name="Matformer")
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        """

        graphs = [PygGraph.atom_dgl_multigraph(s,
                                               neighbor_strategy=config.neighbor_strategy,
                                               cutoff=config.cutoff,
                                               atom_features="atomic_number",
                                               max_neighbors=config.max_neighbors,
                                               use_angle=config.use_angle,
                                               use_lattice=config.use_lattice,
                                               use_canonize=config.use_canonize,
                                               compute_line_graph=False) for s in tqdm(structures)]
        self.graphs = graphs
        self.target = targets
        self.line_graph = config.line_graph

        self.labels = torch.tensor(targets).type(
            torch.get_default_dtype()
        )

        self.transform = config.transform

        features = self._get_attribute_lookup(config.atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for g in graphs:
            z = g.x
            g.atomic_number = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.x.size(0) == 1:
                f = f.unsqueeze(0)
            g.x = f

        self.prepare_batch = prepare_pyg_batch
        self.collate_fn = self.collate_fn_g
        if config.line_graph:
            self.collate_fn = self.collate_fn_lg
            self.prepare_batch = prepare_pyg_line_graph_batch
            print("building line graphs")
            print(self.line_graph)
            if config.lineControl == False:
                self.line_graphs = []
                self.graphs = []
                for g in tqdm(graphs):
                    linegraph_trans = LineGraph(force_directed=True)
                    g_new = Data()
                    g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
                    try:
                        lg = linegraph_trans(g)
                    except Exception as exp:
                        print(g.x, g.edge_attr, exp)
                        pass
                    lg.edge_attr = pyg_compute_bond_cosines(lg)  # old cosine emb
                    # lg.edge_attr = pyg_compute_bond_angle(lg)
                    self.graphs.append(g_new)
                    self.line_graphs.append(lg)
            else:
                if config.neighbor_strategy == "pairwise-k-nearest":
                    self.graphs = []
                    labels = []
                    idx_t = 0
                    filter_out = 0
                    max_size = 0
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        if g.x.size(0) > max_size:
                            max_size = g.x.size(0)
                        if g.x.size(0) < 200:
                            self.graphs.append(g)
                            labels.append(self.labels[idx_t])
                        else:
                            filter_out += 1
                        idx_t += 1
                    print(
                        "filter out %d samples because of exceeding threshold of 200 for nn based method" % filter_out)
                    print("dataset max atom number %d" % max_size)
                    self.line_graphs = self.graphs
                    self.labels = labels
                    self.labels = torch.tensor(self.labels).type(
                        torch.get_default_dtype()
                    )
                else:
                    self.graphs = []
                    for g in tqdm(graphs):
                        g.edge_attr = g.edge_attr.float()
                        self.graphs.append(g)
                    self.line_graphs = self.graphs

        if config.classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        if atom_features == "mat2vec":
            id_prop_file = "mat2vec.csv"
            path = Path(__file__).parent.parent / 'data' / id_prop_file
            return pd.read_csv(path).to_numpy()[:, 1:].astype("float32")
        else:
            template = get_node_attributes("C", atom_features)
            features = np.zeros((1 + max_z, len(template)))

            for element, v in chem_data.items():
                z = v["Z"]
                x = get_node_attributes(element, atom_features)

                if x is not None:
                    features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        if self.line_graph:
            return g, self.line_graphs[idx], label, label

        return g, label

    def setup_standardizer(self, ids):
        """Atom-wise feature standardization transform."""
        x = torch.cat(
            [
                g.x
                for idx, g in enumerate(self.graphs)
                if idx in ids
            ]
        )
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = PygStandardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate_fn_g(samples: List[Tuple[Data, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_fn_lg(
            samples: List[Tuple[Data, Data, torch.Tensor, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, lattice, labels = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        batched_line_graph = Batch.from_data_list(line_graphs)
        if len(labels[0].size()) > 0:
            return (batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice])), torch.stack(labels)
        else:
            return (batched_graph, batched_line_graph, torch.cat([i.unsqueeze(0) for i in lattice])), torch.tensor(labels)


def canonize_edge(
        src_id,
        dst_id,
        src_image,
        dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges_submit(
        atoms=None,
        cutoff=8,
        max_neighbors=12,
        id=None,
        use_canonize=False,
        use_lattice=False,
        use_angle=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    lat = atoms.lattice
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1
        return nearest_neighbor_edges_submit(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )

    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

        if use_lattice:
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 0])))
            edges[(site_idx, site_idx)].add(tuple(np.array([0, 1, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 0, 1])))
            edges[(site_idx, site_idx)].add(tuple(np.array([1, 1, 0])))

    return edges


def pair_nearest_neighbor_edges(
        atoms=None,
        pair_wise_distances=6,
        use_lattice=False,
        use_angle=False,
):
    """Construct pairwise k-fully connected edge list."""
    smallest = pair_wise_distances
    lattice_list = torch.as_tensor(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]).float()

    lattice = torch.as_tensor(atoms.lattice_mat).float()
    pos = torch.as_tensor(atoms.cart_coords)
    atom_num = pos.size(0)
    lat = atoms.lattice
    radius_needed = min(lat.a, lat.b, lat.c) * (smallest / 2 - 1e-9)
    r_a = (np.floor(radius_needed / lat.a) + 1).astype(np.int)
    r_b = (np.floor(radius_needed / lat.b) + 1).astype(np.int)
    r_c = (np.floor(radius_needed / lat.c) + 1).astype(np.int)
    period_list = np.array([l for l in itertools.product(
        *[list(range(-r_a, r_a + 1)), list(range(-r_b, r_b + 1)), list(range(-r_c, r_c + 1))])])
    period_list = torch.as_tensor(period_list).float()
    n_cells = period_list.size(0)
    offset = torch.matmul(period_list, lattice).view(n_cells, 1, 3)
    expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offset).transpose(0, 1).contiguous()
    dist = (pos.unsqueeze(1).unsqueeze(1) - expand_pos.unsqueeze(
        0))  # [n, 1, 1, 3] - [1, n, n_cell, 3] -> [n, n, n_cell, 3]
    dist2, index = torch.sort(dist.norm(dim=-1), dim=-1, stable=True)
    max_value = dist2[:, :, smallest - 1]  # [n, n]
    mask = (dist.norm(dim=-1) <= max_value.unsqueeze(-1))  # [n, n, n_cell]
    shift = torch.matmul(lattice_list, lattice).repeat(atom_num, 1)
    shift_src = torch.arange(atom_num).unsqueeze(-1).repeat(1, lattice_list.size(0))
    shift_src = torch.cat([shift_src[i, :] for i in range(shift_src.size(0))])

    indices = torch.where(mask)
    dist_target = dist[indices]
    u, v, _ = indices
    if use_lattice:
        u = torch.cat((u, shift_src), dim=0)
        v = torch.cat((v, shift_src), dim=0)
        dist_target = torch.cat((dist_target, shift), dim=0)
        assert u.size(0) == dist_target.size(0)

    return u, v, dist_target


def build_undirected_edgedata(
        atoms=None,
        edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():

        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            # if np.linalg.norm(d)!=0:
            # print ('jv',dst_image,d)
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())

    return u, v, r


class PygGraph(object):
    """Generate a graph object."""

    def __init__(
            self,
            nodes=[],
            node_attributes=[],
            edges=[],
            edge_attributes=[],
            color_map=None,
            labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
            atoms=None,
            neighbor_strategy="k-nearest",
            cutoff=8.0,
            max_neighbors=12,
            atom_features="cgcnn",
            max_attempts=3,
            id: Optional[str] = None,
            compute_line_graph: bool = True,
            use_canonize: bool = False,
            use_lattice: bool = False,
            use_angle: bool = False,
    ):
        if neighbor_strategy == "k-nearest":
            edges = nearest_neighbor_edges_submit(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
            u, v, r = build_undirected_edgedata(atoms, edges)
        elif neighbor_strategy == "pairwise-k-nearest":
            u, v, r = pair_nearest_neighbor_edges(
                atoms=atoms,
                pair_wise_distances=2,
                use_lattice=use_lattice,
                use_angle=use_angle,
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        edge_index = torch.cat((u.unsqueeze(0), v.unsqueeze(0)), dim=0).long()
        g = Data(x=node_features, edge_index=edge_index, edge_attr=r)

        if compute_line_graph:
            linegraph_trans = LineGraph(force_directed=True)
            g_new = Data()
            g_new.x, g_new.edge_index, g_new.edge_attr = g.x, g.edge_index, g.edge_attr
            lg = linegraph_trans(g)
            lg.edge_attr = pyg_compute_bond_cosines(lg)
            return g_new, lg
        else:
            return g


def pyg_compute_bond_cosines(lg):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
            torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


def pyg_compute_bond_angle(lg):
    """Compute bond angle from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    src, dst = lg.edge_index
    x = lg.x
    r1 = -x[src]
    r2 = x[dst]
    a = (r1 * r2).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(r1, r2).norm(dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return angle


class PygStandardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: Data):
        """Apply standardization to atom_features."""
        h = g.x
        g.x = (h - self.mean) / self.std
        return g


def prepare_pyg_batch(
        batch: Tuple[Data, torch.Tensor], device=None, non_blocking=False, subset=None
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    if subset is not None:
        return (
            Batch.from_data_list(g.index_select(list(range(subset)))).to(device=device, non_blocking=non_blocking),
            t[:subset].to(device=device, non_blocking=non_blocking)
        )

    return (
        g.to(device, non_blocking=non_blocking),
        t.to(device, non_blocking=non_blocking),
    )


def prepare_pyg_line_graph_batch(
        batch: Tuple[Tuple[Data, Data, torch.Tensor], torch.Tensor],
        device=None,
        non_blocking=False,
        subset=None
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, lattice = batch[0]

    t = batch[1]
    if subset is not None:
        return (
            Batch.from_data_list(g.index_select(list(range(subset)))).to(device=device, non_blocking=non_blocking),
            Batch.from_data_list(lg.index_select(list(range(subset)))).to(device=device, non_blocking=non_blocking),
            lattice[:subset].to(device, non_blocking=non_blocking),
            t[:subset].to(device=device, non_blocking=non_blocking)
        )

    batch = (
        (
            g.to(device, non_blocking=non_blocking),
            lg.to(device, non_blocking=non_blocking),
            lattice.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


if __name__ == "__main__":
    from jarvis.db.figshare import data
    from jarvis.core.atoms import Atoms

    d = data('dft_2d')
    atoms = [datum['atoms'] for datum in d]
