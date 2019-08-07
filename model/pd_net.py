from src.layers import *


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