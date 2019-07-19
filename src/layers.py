import torch
from torch_geometric.nn.models import GAE
import numpy as np
import scipy
from torch.nn import Parameter as Param
from torch import Tensor
from torch_geometric.nn.conv import RGCNConv, GCNConv, MessagePassing
from sklearn import metrics
import torch.nn.functional as F
from torch_geometric.data import Data


torch.manual_seed(1111)


EPS = 1e-13


def negative_sampling(pos_edge_index, num_nodes):
    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    perm = torch.tensor(np.random.choice(num_nodes**2, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(np.random.choice(num_nodes**2, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).to(pos_edge_index.device)


def sparse_id(n):
    idx = [[i for i in range(n)], [i for i in range(n)]]
    val = [1 for i in range(n)]
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(val)
    shape = (n, n)

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def auprc_auroc_ap(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    auroc, ap = metrics.roc_auc_score(y, pred), metrics.average_precision_score(y, pred)
    y, xx, _ = metrics.ranking.precision_recall_curve(y, pred)
    auprc = metrics.ranking.auc(xx, y)

    return auprc, auroc, ap


def uniform(size, tensor):
    bound = 1.0 / np.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class MyRGCNConv(MessagePassing):
    r"""
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases,
                 after_relu,
                 bias=True,
                 **kwargs):
        super(MyRGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.after_relu = after_relu

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        self.root = Param(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type):
        """"""
        return self.propagate(
            edge_index, x=x, edge_type=edge_type)

    def message(self, x_j, edge_index_j, edge_type):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.

        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        w = torch.index_select(w, 0, edge_type)
        out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out

    def update(self, aggr_out, x):

        out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)