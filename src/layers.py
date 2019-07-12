import torch
from torch_geometric.nn.models import GAE
import numpy as np
import scipy
from torch.nn import Parameter
from torch import Tensor
from torch_geometric.nn.conv import RGCNConv, GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models.autoencoder import negative_sampling


torch.manual_seed(1111)

def sparse_id(n):
    idx = [[i for i in range(n)], [i for i in range(n)]]
    val = [1 for i in range(n)]
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(val)
    shape = (n, n)

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))