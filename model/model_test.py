from data.utils import load_data_torch, apk
from model.ddip import HGCN
from torch_geometric.data import Data
from torch_geometric.nn.models.autoencoder import negative_sampling
import torch as t
from torch.nn.functional import binary_cross_entropy
import numpy as np
from sklearn import metrics
import pickle


# ##############################################
# load data
et_list = [0, 1]
data = load_data_torch("/Users/nyxfer/Docu/FM-PSEP/data/", et_list, mono=True)


[n_drug, n_feat_d] = data['d_feat'].shape
n_et_dd = len(et_list)


# ##############################################
print('construct feed dictionary')
feed_dict = {
    'd_feat': data['d_feat'],
    'dd_edge_index': data['dd_edge_index'],
    'dd_edge_type': data['dd_edge_type'],
    'dd_edge_type_num': data['dd_edge_type_num'],
    'dd_y': data['dd_y']
}

da = Data.from_dict(feed_dict)

print('constructing model, opt and sent data to device')
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = HGCN(n_feat_d, n_et_dd , n_et_dd).to(device)
optimizer = t.optim.Adam(model.parameters())
da = da.to(device)


# train
def train():
    model.train()

    optimizer.zero_grad()
    out = model(da.d_feat, da.dd_edge_index, da.dd_edge_type, da.dd_edge_type_num)

    loss = binary_cross_entropy(out, data['dd_y'])
    loss.backward()
    optimizer.step()
    return out, loss


# acc
def get_acc(pred, targ):
    pred = pred > 0.5
    roc_sc = metrics.roc_auc_score(targ.tolist(), pred.tolist())
    aupr_sc = metrics.average_precision_score(targ.tolist(), pred.tolist())
    # apk_sc = apk(targ.tolist(), pred.tolist(), k=50)

    return roc_sc, aupr_sc


print('model training ...')
for epoch in range(1, 100):
    o, l = train()
    roc, aupr = get_acc(o, data['dd_y'])
    log = 'Epoch: {:03d}, Train loss: {:.4f}, Train roc: {:.4f}, train prc: {:.4f}'
    print(log.format(epoch, l, roc, aupr))