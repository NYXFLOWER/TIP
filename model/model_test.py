from data.utils import load_data_torch
from model.ddip import HGCN
from torch_geometric.data import Data
from torch_geometric.nn.models.autoencoder import negative_sampling
import torch as t
from torch.nn.functional import binary_cross_entropy
import numpy as np
from sklearn import metrics
import pickle
import time
from tempfile import TemporaryFile



# ##############################################
# load data
with open("/Users/nyxfer/Docu/FM-PSEP/data/training_samples_500.pkl", "rb") as f:
    et_list = pickle.load(f)
et_list = et_list
feed_dict = load_data_torch("/Users/nyxfer/Docu/FM-PSEP/data/", et_list, mono=True)


[n_drug, n_feat_d] = feed_dict['d_feat'].shape
n_et_dd = len(et_list)


data = Data.from_dict(feed_dict)

print('constructing model, opt and sent data to device')
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = HGCN(n_feat_d, n_et_dd, 50).to(device)
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
data = data.to(device)


# train
def train():
    model.train()

    optimizer.zero_grad()
    pos_pred, neg_pred = model(data.d_feat, data.dd_edge_index, data.dd_edge_type, data.dd_edge_type_num)

    pos_loss = binary_cross_entropy(pos_pred, data.dd_y_pos)
    neg_loss = binary_cross_entropy(neg_pred, data.dd_y_neg)
    (pos_loss).backward()

    optimizer.step()

    return pos_pred, neg_pred, pos_loss, neg_loss


# acc
def get_acc(pred, targ):
    pred = pred > 0.5
    roc_sc = metrics.roc_auc_score(targ.tolist(), pred.tolist())
    aupr_sc = metrics.average_precision_score(targ.tolist(), pred.tolist())
    # apk_sc = apk(targ.tolist(), pred.tolist(), k=50)


    return roc_sc, aupr_sc


EPOCH_NUM = 100
train_loss = np.zeros(EPOCH_NUM)
# y_targ = t.cat((data.dd_y_pos, data.dd_y_neg), 0)


print('model training ...')
for epoch in range(EPOCH_NUM):
    start = time.time()
    p_pred, n_pred, p_loss, n_loss = train()


    # acc pos
    p_pred = p_pred > 0.5
    pos_acc = p_pred.sum().tolist() / data.dd_edge_type_num[-1]
    # acc neg
    n_pred = n_pred <= 0.5
    neg_acc = n_pred.sum().tolist() / data.dd_edge_type_num[-1]

    log = 'Epoch: {:03d}/{:.2f}, pos_loss: {:.4f}, neg_loss: {:.4f}, total_loss: {:.4f}, pos_acc: {:.4f}, neg_acc: {:.4f}'
    train_loss[epoch] = p_loss+n_loss
    print(log.format(epoch, time.time() - start, p_loss, n_loss, train_loss[epoch], pos_acc, neg_acc))


outfile = TemporaryFile()
np.save(outfile, train_loss)

# outfile.seek(0) # Only needed here to simulate closing & reopening file
# np.load(outfile)
