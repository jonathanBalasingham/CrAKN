"""Torch Module for E(n) Equivariant Graph Convolutional Layer"""
import gettext

# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch.nn as nn
from dgl import function as fn
from dgl import softmax_edges


class EGNNConv(nn.Module):
    r"""Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`__

    .. math::

        m_{ij}=\phi_e(h_i^l, h_j^l, ||x_i^l-x_j^l||^2, a_{ij})

        x_i^{l+1} = x_i^l + C\sum_{j\in\mathcal{N}(i)}(x_i^l-x_j^l)\phi_x(m_{ij})

        m_i = \sum_{j\in\mathcal{N}(i)} m_{ij}

        h_i^{l+1} = \phi_h(h_i^l, m_i)

    where :math:`h_i`, :math:`x_i`, :math:`a_{ij}` are node features, coordinate
    features, and edge features respectively. :math:`\phi_e`, :math:`\phi_h`, and
    :math:`\phi_x` are two-layer MLPs. :math:`C` is a constant for normalization,
    computed as :math:`1/|\mathcal{N}(i)|`.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in the two-layer MLPs in
        :math:`\phi_e, \phi_x, \phi_h`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.

    """

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size)
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False)
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [edges.src['h'], edges.dst['h'], edges.data['radial'], edges.data['a']],
                dim=-1
            )
        else:
            f = torch.cat([edges.src['h'], edges.dst['h'], edges.data['radial']], dim=-1)

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data['x_diff']

        return {'msg_x': msg_x, 'msg_h': msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        coord_feat : torch.Tensor
            The coordinate feature of shape :math:`(N, h_x)`. :math:`N` is the
            number of nodes, and :math:`h_x` can be any positive integer.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        node_feat_out : torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            is the same as out_size.
        coord_feat_out: torch.Tensor
            The output coordinate feature of shape :math:`(N, h_x)` where :math:`h_x`
            is the same as the input coordinate feature dimension.
        """
        with graph.local_scope():
            # node feature
            graph.ndata['h'] = node_feat
            # coordinate feature
            graph.ndata['x'] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata['a'] = edge_feat

            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v('x', 'x', 'x_diff'))
            graph.edata['radial'] = graph.edata['x_diff'].square().sum(dim=1).unsqueeze(-1)

            # normalize coordinate difference
            graph.edata['x_diff'] = graph.edata['x_diff'] / ((graph.edata['radial'] + 1e-8).sqrt())

            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e('msg_x', 'm'), fn.mean('m', 'x_neigh'))
            graph.update_all(fn.copy_e('msg_h', 'm'), fn.sum('m', 'h_neigh'))

            h_neigh, x_neigh = graph.ndata['h_neigh'], graph.ndata['x_neigh']

            h = self.node_mlp(
                torch.cat([node_feat, h_neigh], dim=-1)
            )
            x = coord_feat + x_neigh

            return h, x


class CrAKNConvV1(nn.Module):

    def __init__(self,
                 embedding_dim,
                 edge_feat_size=0,
                 attention_dropout=0.0,
                 qkv_bias=True,
                 use_multiplier=True,
                 use_bias=True,
                 activation=nn.Mish):
        super(CrAKNConvV1, self).__init__()

        self.embedding_dim = embedding_dim
        self.use_multiplier = use_multiplier
        self.use_bias = use_bias
        self.edge_feat_size = edge_feat_size

        self.linear_q = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias),
            nn.LayerNorm(embedding_dim),
            activation(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias),
            nn.LayerNorm(embedding_dim),
            activation(inplace=True),
        )

        self.linear_v = nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias)

        if self.use_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                activation(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
            )
        if self.use_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                activation(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
            )

        self.weight_encoding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            activation(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.node_mlp = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim, embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, embedding_dim, bias=False)
        )

    def message(self, edges):
        relation_qk = edges.data['relative_qk']
        if self.use_multiplier:
            pem = self.linear_p_multiplier(edges.dst['x'])
            relation_qk = relation_qk * pem

        if self.use_bias:
            peb = self.linear_p_bias(edges.dst['x'])
            relation_qk = relation_qk + peb

        weight = self.weight_encoding(relation_qk)
        return {'weight': weight}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):

        with graph.local_scope():
            # node feature
            graph.ndata['h'] = node_feat
            # coordinate feature
            graph.ndata['x'] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata['a'] = edge_feat

            graph.ndata['q'] = self.linear_q(graph.ndata['x'])
            graph.ndata['k'] = self.linear_k(graph.ndata['x'])
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v('q', 'k', 'relative_qk'))
            graph.edata['radial'] = graph.edata['relative_qk'].square().sum(dim=1).unsqueeze(-1)

            # normalize coordinate difference
            graph.edata['relative_qk'] = graph.edata['relative_qk'] / ((graph.edata['radial'] + 1e-8).sqrt())

            graph.ndata['v'] = self.linear_v(node_feat)

            graph.apply_edges(self.message)
            graph.edata['weight'] = softmax_edges(graph, 'weight')
            graph.update_all(fn.copy_e('weight', 'm'), fn.sum('m', 'h_neigh'))

            h_neigh = graph.ndata['h_neigh']

            h = self.node_mlp(
                torch.cat([node_feat, h_neigh], dim=-1)
            )

            x = self.coord_mlp(coord_feat)

            return h, x



class CrAKNConvV2(nn.Module):

    def __init__(self,
                 embedding_dim,
                 edge_keys,
                 edge_feat_size=0,
                 attention_dropout=0.0,
                 qkv_bias=True,
                 use_multiplier=True,
                 use_bias=True,
                 activation=nn.Mish):
        super(CrAKNConvV2, self).__init__()

        self.embedding_dim = embedding_dim
        self.use_multiplier = use_multiplier
        self.use_bias = use_bias
        self.edge_feat_size = edge_feat_size
        self.num_edge_types = len(edge_keys)
        self.edge_keys = edge_keys

        self.linear_q = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias),
            nn.LayerNorm(embedding_dim),
            activation(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias),
            nn.LayerNorm(embedding_dim),
            activation(inplace=True),
        )

        self.linear_v = nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias)

        if self.use_multiplier:
            self.linear_p_multiplier = nn.ModuleDict({
                k: nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    activation(inplace=True),
                    nn.Linear(embedding_dim, embedding_dim),
                ) for k in self.edge_keys
            })
        if self.use_bias:
            self.linear_p_bias = nn.ModuleDict({
                k: nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    activation(inplace=True),
                    nn.Linear(embedding_dim, embedding_dim),
                ) for k in self.edge_keys
            })

        self.weight_encoding = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                activation(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
            ) for k in self.edge_keys
        })

        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.node_mlp = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim, embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, embedding_dim, bias=False)
        )

        self.combine = nn.Sequential(
            nn.Linear(embedding_dim * self.num_edge_types, embedding_dim * self.num_edge_types),
            nn.Mish(),
            nn.Linear(embedding_dim * self.num_edge_types, embedding_dim, bias=False)
        )

    def message(self, edges, ekey):
        relation_qk = edges.data['relative_qk']
        if self.use_multiplier:
            pem = self.linear_p_multiplier[ekey](edges.data[ekey])
            relation_qk = relation_qk * pem

        if self.use_bias:
            peb = self.linear_p_bias[ekey](edges.data[ekey])
            relation_qk = relation_qk + peb

        weight = self.weight_encoding[ekey](relation_qk)
        return {f'value_{ekey}': weight * edges.dst['v']}

    def update(self, edges, ekey):

        return {f'value_{ekey}': edges.dst['v'] * edges.data[f'weight_{ekey}']}

    def forward(self, graph, node_feat):

        with graph.local_scope():
            graph.ndata['x'] = node_feat

            graph.ndata['q'] = self.linear_q(graph.ndata['x'])
            graph.ndata['k'] = self.linear_k(graph.ndata['x'])
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v('q', 'k', 'relative_qk'))
            graph.edata['radial'] = graph.edata['relative_qk'].square().sum(dim=1).unsqueeze(-1)

            # normalize coordinate difference
            graph.edata['relative_qk'] = graph.edata['relative_qk'] / ((graph.edata['radial'] + 1e-8).sqrt())
            graph.ndata['v'] = self.linear_v(node_feat)

            for ekey in self.edge_keys:
                graph.apply_edges(lambda e: self.message(e, ekey))
                #graph.edata[f'weight_{ekey}'] = softmax_edges(graph, f'weight_{ekey}')
                #graph.apply_edges(lambda e: self.update(e, ekey))
                graph.update_all(fn.copy_e(f'value_{ekey}', f'm_{ekey}'), fn.sum(f'm_{ekey}', f'h_neigh_{ekey}'))

            h = self.combine(
                torch.cat([graph.ndata[f'h_neigh_{key}'] for key in self.edge_keys], dim=-1)
            )

            return h
