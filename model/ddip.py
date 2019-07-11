import torch as t
import numpy as np
import scipy.sparse as sp
from torch.nn import Module, Linear, functional as F, Embedding
from torch_geometric.nn import RGCNConv

EMBEDDING_DIM = 10
LAYER_DD = 8
LAYER_DD_OUT = 6


class HGCN(Module):
    def __init__(self, in_dim, num_et, num_base):
        super(HGCN, self).__init__()
        self.num_et = num_et

        # encoder layers
        self.embedding = Linear(in_dim, EMBEDDING_DIM)
        self.rgcn1 = RGCNConv(EMBEDDING_DIM, LAYER_DD_OUT, num_et, num_base)

        # decoder layer
        self.rel_embedding = Embedding(num_et, LAYER_DD_OUT)

    def encoder(self, x, edge_index, edge_type):
        x = self.embedding(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.relu(x)
        return x

    def decoder(self, x, edge_index, edge_type_num):
        result = t.tensor([])
        for i in range(self.num_et):
            ind = t.index_select(edge_index, 1, t.tensor(range(edge_type_num[i], edge_type_num[i + 1])))
            m = self.rel_embedding(t.tensor([i], dtype=t.long))
            m = t.diag(m[0])
            # x_et = (x[ind[0]] * m[ind[0]][:, ind[1]]).sum(dim=1)

            m = t.mm(x, m)
            m = t.mm(m, x.t())
            m = t.sigmoid(m)

            ind = ind.tolist()
            result = t.cat((result, m[ind[0], ind[1]]))

        return result

    def forward(self, x, edge_index, edge_type, edge_type_num):
        x = self.encoder(x, edge_index, edge_type)
        x = self.decoder(x, edge_index, edge_type_num)
        return x




