import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
from torch_geometric.nn import GCNConv, RGCNConv
from data.utils import load_data

# #######################
# parameter setting
P_DIM_1 = 10
P_DIM_2 = 5
P_DIM_PD = 4
D_DIM_EMBED = 6
D_DIM_DD1 = 7
D_DIM_OUT = 5


# #######################
# load data
et_list = [0, 1, 2]
data = load_data("./data/", et_list, mono=False)

n_feat_p = data["p_feat"].shape[1]
n_feat_d = data['d_feat'].shape[1]
n_et_dd = len(et_list)


# #######################
# mini batch
class Batch:
    def __init__(self):
        self.pp_edge_ind = None
        self.pd_edge_ind = None
        self.dd_edge_ind = None
        self.dd_edge_type = None


batch_bank = [Batch()]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_p1 = GCNConv(n_feat_p, P_DIM_1)
        self.conv_p2 = GCNConv(P_DIM_1, P_DIM_2)

        self.rconv_pd = RGCNConv(P_DIM_2, P_DIM_PD, 1, 1)
        self.embed_d = Linear(n_feat_d, D_DIM_EMBED)

        self.rconv_dd1 = RGCNConv((D_DIM_EMBED + P_DIM_PD), D_DIM_DD1, n_et_dd, n_et_dd)
        self.rconv_dd2 = RGCNConv(D_DIM_DD1, D_DIM_OUT, n_et_dd, n_et_dd)

    def forward(self, batch_data):
        p = F.relu(self.conv_p1(data['p_feat'], batch_data.pp_edge_ind))
        p = F.relu(self.conv_p2(p, batch_data.pp_edge_ind))

        pd_edge_type = torch.tensor(np.zeros(batch_data.pd_edge_ind.shape[1]))
        p = F.relu(self.rconv_pd(p, batch_data.pd_edge_ind, pd_edge_type))
        d = F.relu(self.embed_d(data['d_feat']))
        d = torch.cat((d, p), 1)

        d = F.relu(self.rconv_dd1(d, batch_data.dd_edge_ind, batch_data.dd_edge_type))
        d = self.rconv_dd2(d, batch_data.dd_edge_ind, batch_data.dd_edge_type)

        return d


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 26:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for batch in batch_bank:
        data = data.to(device)
        optimizer.zero_grad()
        F.nll_loss(model(data), data.y).backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0

    # for data in test_loader:
    #     data = data.to(device)
    #     pred = model(data).max(1)[1]
    #     correct += pred.eq(data.y).sum().item()
    # return correct / len(test_dataset)


for epoch in range(1, 31):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))