from src.layers import *


class MyPDConv(MessagePassing):
    """ directed gcn layer for pd-net """
    def __init__(self, in_dim, out_dim, is_after_relu=True, is_bias=False,
                 **kwargs):
        super(MyPDConv, self).__init__(aggr='mean', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
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

    def update(self, aggr_out, x):
        out = torch.mm(aggr_out, self.weight)

        if self.bias:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}'.format(self.__class__.__name__,
                                  self.in_dim,
                                  self.out_dim)