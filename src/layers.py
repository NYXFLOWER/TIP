import torch
from torch_geometric.nn.models import GAE, InnerProductDecoder
import numpy as np
import scipy.sparse as sp
from torch.nn import Parameter as Param
from torch import Tensor
from torch_geometric.nn.conv import RGCNConv, GCNConv, MessagePassing
from sklearn import metrics
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch_geometric.data import Data
from pytorch_memlab import profile
from src.neg_sampling import typed_negative_sampling, negative_sampling


torch.manual_seed(1111)
np.random.seed(1111)
EPS = 1e-13


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
                 bias=False,
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

        self.att.data.normal_(std=1/np.sqrt(self.num_bases))

        if self.after_relu:
            self.root.data.normal_(std=2/self.in_channels)
            self.basis.data.normal_(std=2/self.in_channels)

        else:
            self.root.data.normal_(std=1/np.sqrt(self.in_channels))
            self.basis.data.normal_(std=1/np.sqrt(self.in_channels))

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, edge_index, edge_type):
        """"""
        return self.propagate(
            edge_index, x=x, edge_type=edge_type)

    def message(self, x_j, edge_index_j, edge_type):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        w = w[edge_type, :, :]
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


class MyRGCNConv2(MessagePassing):
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
                 bias=False,
                 **kwargs):
        super(MyRGCNConv2, self).__init__(aggr='mean', **kwargs)

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

        self.att.data.normal_(std=1/np.sqrt(self.num_bases))

        if self.after_relu:
            self.root.data.normal_(std=2/self.in_channels)
            self.basis.data.normal_(std=2/self.in_channels)

        else:
            self.root.data.normal_(std=1/np.sqrt(self.in_channels))
            self.basis.data.normal_(std=1/np.sqrt(self.in_channels))

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, edge_index, edge_type, range_list):
        """"""
        return self.propagate(
            edge_index, x=x, edge_type=edge_type, range_list=range_list)

    def message(self, x_j, edge_index, edge_type, range_list):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        # w = w[edge_type, :, :]
        # out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        out_list = []
        for et in range(range_list.shape[0]):
            start, end = range_list[et]

            tmp = torch.matmul(x_j[start: end, :], w[et])

            # xxx = x_j[start: end, :]
            # tmp = checkpoint(torch.matmul, xxx, w[et])

            out_list.append(tmp)

        # TODO: test this
        return torch.cat(out_list)

    def update(self, aggr_out, x):

        out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


class MyGAE(torch.nn.Module):

    def __init__(self, encoder, decoder=None):
        super(MyGAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder


class MyHierarchyConv(MessagePassing):
    """ directed gcn layer for pd-net """
    def __init__(self, in_dim, out_dim,
                 unigue_source_num, unique_target_num,
                 is_after_relu=True, is_bias=False, **kwargs):

        super(MyHierarchyConv, self).__init__(aggr='mean', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.unique_source_num = unigue_source_num
        self.unique_target_num = unique_target_num
        self.is_after_relu = is_after_relu

        # parameter setting
        self.weight = Param(torch.Tensor(in_dim, out_dim))

        if is_bias:
            self.bias = Param(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.is_after_relu:
            self.weight.data.normal_(std=1/np.sqrt(self.in_dim))
        else:
            self.weight.data.normal_(std=2/np.sqrt(self.in_dim))

        if self.bias:
            self.bias.data.zero_()

    def forward(self, x, edge_index, range_list):
        return self.propagate(edge_index, x=x, range_list=range_list)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, range_list):
        if self.bias:
            aggr_out += self.bias

        out = torch.matmul(aggr_out[self.unique_source_num:, :], self.weight)
        assert out.shape[0] == self.unique_target_num

        return out

    def __repr__(self):
        return '{}({}, {}'.format(self.__class__.__name__,
                                  self.in_dim,
                                  self.out_dim)