from src.layers import *


class encoder(torch.nn.Module):
    def __init__(self, p_input_dim, d_input_dim, p_hid1_dim, p_out_dim, d_emb_dim,
                 n_r, n_base, d_hid1_dim, d_out_dim):
        super(encoder, self).__init__()

        # p-net
        self.p_conv1 = GCNConv(p_input_dim, p_hid1_dim, bias=False)

        # pd-net
        # TODO: remove self-loop
        self.pd_conv = GCNConv(p_hid1_dim, p_out_dim, bias=False)

        # d-net
        self.d_embedding = Parameter(Tensor(d_input_dim, d_emb_dim))
        self.d_conv1 = RGCNConv(p_out_dim + d_emb_dim, d_hid1_dim,
                                n_r, n_base, bias=False)
        self.d_conv2 = RGCNConv(d_hid1_dim, d_out_dim, n_r, n_base, bias=False)

    def forward(self, x_p, x_d, edge_index_p, edge_index_pd,
                edge_index_d, edge_type, edge_norm):
        # p-net
        x_p = F.relu(self.p_conv1(x_p, edge_index_p))

        # pd-net
        x_d1 = self.pd_conv(x_p, edge_index_pd)

        # d-embedding
        x_d2 = torch.matmul(x_d, self.d_embedding)

        # d-net
        x_d = torch.cat(x_d1, x_d2, dim=1)
        x_d = F.relu(self.d_conv1(x_d, edge_index_d, edge_type))
        x_d = self.d_conv2(x_d, edge_index_d, edge_type)

        return x_d


