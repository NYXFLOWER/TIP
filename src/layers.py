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


#########################################################################
# Convolutional layer
#########################################################################
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


#########################################################################
# Training Framework
#########################################################################
class MyGAE(torch.nn.Module):

    def __init__(self, encoder, decoder=None):
        super(MyGAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder


#########################################################################
# Encoder - Node Representation Learning
#########################################################################
class PPEncoder(torch.nn.Module):

    def __init__(self, in_dim, hid1=32, hid2=16):
        super(PPEncoder, self).__init__()
        self.out_dim = hid2

        self.conv1 = GCNConv(in_dim, hid1, cached=True)
        self.conv2 = GCNConv(hid1, hid2, cached=True)

        # self.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x, inplace=True)
        x = self.conv2(x, edge_index)
        return x

    # def reset_parameters(self):
    #     self.embed.data.normal_()


class FMEncoderCat(torch.nn.Module):

    def __init__(self, device, in_dim_drug, num_dd_et, in_dim_prot,
                 uni_num_prot, uni_num_drug, prot_drug_dim=16,
                 num_base=32, n_embed=48, n_hid1=32, n_hid2=16):
        '''
        :param device:
        :param in_dim_drug:
        :param num_dd_et:
        :param in_dim_prot:
        :param uni_num_prot:
        :param uni_num_drug:
        :param prot_drug_dim:
        :param num_base:
        :param n_embed:
        :param n_hid1:
        :param n_hid2:
        :param mod: 'cat', 'ave'
        '''
        super(FMEncoderCat, self).__init__()
        self.num_et = num_dd_et
        self.out_dim = n_hid2
        self.uni_num_drug = uni_num_drug
        self.uni_num_prot = uni_num_prot

        # on pp-net
        self.pp_encoder = PPEncoder(in_dim_prot)

        # feat: drug index
        self.embed = Param(torch.Tensor(in_dim_drug, n_embed))

        # on pd-net
        self.hgcn = MyHierarchyConv(self.pp_encoder.out_dim, prot_drug_dim, uni_num_prot, uni_num_drug)
        self.hdrug = torch.zeros((self.uni_num_drug, self.pp_encoder.out_dim)).to(device)

        # on dd-net
        self.rgcn1 = MyRGCNConv2(n_embed+self.hgcn.out_dim, n_hid1, num_dd_et, num_base, after_relu=False)
        self.rgcn2 = MyRGCNConv2(n_hid1, n_hid2, num_dd_et, num_base, after_relu=True)

        self.reset_parameters()

    def forward(self, x_drug, dd_edge_index, dd_edge_type, dd_range_list, d_norm, x_prot, pp_edge_index, dp_edge_index, dp_range_list):
        # pp-net
        x_prot = self.pp_encoder(x_prot, pp_edge_index)
        # x_prot = checkpoint(self.pp_encoder, x_prot, pp_edge_index)
        # pd-net
        x_prot = torch.cat((x_prot, self.hdrug))
        x_prot = self.hgcn(x_prot, dp_edge_index, dp_range_list)
        # x_prot = checkpoint(self.hgcn, torch.cat((x_prot, self.hdrug)), dp_edge_index, dp_range_list)

        # d-embed
        x_drug = torch.matmul(x_drug, self.embed)
        # x_drug = checkpoint(torch.matmul, x_drug, self.embed)
        x_drug = x_drug / d_norm.view(-1, 1)
        x_drug = torch.cat((x_drug, x_prot), dim=1)

        # dd-net
        # x = self.rgcn1(x, edge_index, edge_type, range_list)
        # x_drug = checkpoint(self.rgcn1, x_drug, dd_edge_index, dd_edge_type, dd_range_list)
        x_drug = self.rgcn1(x_drug, dd_edge_index, dd_edge_type, dd_range_list)

        x_drug = F.relu(x_drug, inplace=True)
        x_drug = self.rgcn2(x_drug, dd_edge_index, dd_edge_type, dd_range_list)
        # x_drug = checkpoint(self.rgcn2, x_drug, dd_edge_index, dd_edge_type, dd_range_list)
        return x_drug

    def reset_parameters(self):
        self.embed.data.normal_()


class FMEncoder(torch.nn.Module):

    def __init__(self, device, in_dim_drug, num_dd_et, in_dim_prot,
                 uni_num_prot, uni_num_drug, prot_drug_dim=64,
                 num_base=32, n_embed=64, n_hid1=32, n_hid2=16, mod='add'):
        '''
        :param device:
        :param in_dim_drug:
        :param num_dd_et:
        :param in_dim_prot:
        :param uni_num_prot:
        :param uni_num_drug:
        :param prot_drug_dim:
        :param num_base:
        :param n_embed:
        :param n_hid1:
        :param n_hid2:
        :param mod: 'cat', 'ave'
        '''
        super(FMEncoder, self).__init__()
        self.num_et = num_dd_et
        self.out_dim = n_hid2
        self.uni_num_drug = uni_num_drug
        self.uni_num_prot = uni_num_prot

        # on pp-net
        self.pp_encoder = PPEncoder(in_dim_prot)

        # feat: drug index
        self.embed = Param(torch.Tensor(in_dim_drug, n_embed))

        # on pd-net
        self.hgcn = MyHierarchyConv(self.pp_encoder.out_dim, prot_drug_dim, uni_num_prot, uni_num_drug)
        self.hdrug = torch.zeros((self.uni_num_drug, self.pp_encoder.out_dim)).to(device)

        # on dd-net +self.hgcn.out_dim
        self.rgcn1 = MyRGCNConv2(n_embed, n_hid1, num_dd_et, num_base, after_relu=False)
        self.rgcn2 = MyRGCNConv2(n_hid1, n_hid2, num_dd_et, num_base, after_relu=True)

        self.reset_parameters()

    def forward(self, x_drug, dd_edge_index, dd_edge_type, dd_range_list, d_norm,
                x_prot, pp_edge_index, dp_edge_index, dp_range_list):
        # pp-net
        x_prot = self.pp_encoder(x_prot, pp_edge_index)
        # x_prot = checkpoint(self.pp_encoder, x_prot, pp_edge_index)
        # pd-net
        x_prot = torch.cat((x_prot, self.hdrug))
        # x_prot = x_prot + self.hdrug
        x_prot = self.hgcn(x_prot, dp_edge_index, dp_range_list)
        # x_prot = checkpoint(self.hgcn, torch.cat((x_prot, self.hdrug)), dp_edge_index, dp_range_list)

        # d-embed
        x_drug = torch.matmul(x_drug, self.embed)
        # x_drug = checkpoint(torch.matmul, x_drug, self.embed)
        x_drug = x_drug / d_norm.view(-1, 1)
        # x_drug = torch.cat((x_drug, x_prot), dim=1)
        x_drug = x_drug + x_prot

        # dd-net
        # x = self.rgcn1(x, edge_index, edge_type, range_list)
        # x_drug = checkpoint(self.rgcn1, x_drug, dd_edge_index, dd_edge_type, dd_range_list)
        x_drug = self.rgcn1(x_drug, dd_edge_index, dd_edge_type, dd_range_list)

        x_drug = F.relu(x_drug, inplace=True)
        x_drug = self.rgcn2(x_drug, dd_edge_index, dd_edge_type, dd_range_list)
        # x_drug = checkpoint(self.rgcn2, x_drug, dd_edge_index, dd_edge_type, dd_range_list)
        return x_drug

    def reset_parameters(self):
        self.embed.data.normal_()


class HierEncoder(torch.nn.Module):
    def __init__(self, source_dim, embed_dim, target_dim,
                 uni_num_source, uni_num_target):
        super(HierEncoder, self).__init__()

        self.embed = Param(torch.Tensor(source_dim, embed_dim))
        self.hgcn = MyHierarchyConv(embed_dim, target_dim, uni_num_source, uni_num_target)

        self.reset_parameters()

    def reset_parameters(self):
        self.embed.data.normal_()

    def forward(self, source_feat, edge_index, range_list, x_norm):
        x = torch.matmul(source_feat, self.embed)
        x = x / x_norm.view(-1, 1)
        x = self.hgcn(x, edge_index, range_list)
        # x = F.relu(x, inplace=True)

        return x


#########################################################################
# Decoder - Multirelational Link Prediction
#########################################################################
class MultiInnerProductDecoder(torch.nn.Module):
    def __init__(self, in_dim, num_et):
        super(MultiInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Param(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1/np.sqrt(self.in_dim))


class NNDecoder(torch.nn.Module):


    def __init__(self, in_dim, num_uni_edge_type, l1_dim=16):
        """ in_dim: the feat dim of a drug
            num_edge_type: num of dd edge type """

        super(NNDecoder, self).__init__()
        self.l1_dim = l1_dim     # Decoder Lays' dim setting

        # parameters
        # for drug 1
        self.w1_l1 = Param(torch.Tensor(in_dim, l1_dim))
        self.w1_l2 = Param(torch.Tensor(num_uni_edge_type, l1_dim))  # dd_et
        # specified
        # for drug 2
        self.w2_l1 = Param(torch.Tensor(in_dim, l1_dim))
        self.w2_l2 = Param(torch.Tensor(num_uni_edge_type, l1_dim))  # dd_et
        # specified

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type):
        # layer 1
        d1 = torch.matmul(z[edge_index[0]], self.w1_l1)
        d2 = torch.matmul(z[edge_index[1]], self.w2_l1)
        d1 = F.relu(d1, inplace=True)
        d2 = F.relu(d2, inplace=True)

        # layer 2
        d1 = (d1 * self.w1_l2[edge_type]).sum(dim=1)
        d2 = (d2 * self.w2_l2[edge_type]).sum(dim=1)

        return torch.sigmoid(d1 + d2)

    def reset_parameters(self):
        self.w1_l1.data.normal_()
        self.w2_l1.data.normal_()
        self.w1_l2.data.normal_(std=1 / np.sqrt(self.l1_dim))
        self.w2_l2.data.normal_(std=1 / np.sqrt(self.l1_dim))
