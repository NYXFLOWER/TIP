from data.utils import load_data_torch
from src.layers import negative_sampling, auprc_auroc_ap
import pickle
from torch.nn import Module
import torch
from src.layers import *
from model.pd_net import MyHierarchyConv
import sys
import time

sys.setrecursionlimit(8000)

with open('../out/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

# et_list = et_list
feed_dict = load_data_torch("../data/", et_list, mono=True)
data = Data.from_dict(feed_dict)
n_drug, n_drug_feat = data.d_feat.shape
n_prot, n_prot_feat = data.p_feat.shape
n_et_dd = len(et_list)

data.train_idx, data.train_et, data.train_range,data.test_idx, data.test_et, data.test_range = process_edges(data.dd_edge_index)

# re-construct node feature
data.p_feat = torch.cat([dense_id(n_prot), torch.zeros(size=(n_drug, n_prot))], dim=0)
data.d_feat = dense_id(n_drug)
n_drug_feat = n_drug
n_prot_feat = n_prot

# ###################################
# dp_edge_index and range index
# ###################################
data.dp_edge_index = np.array([data.dp_adj.col-1, data.dp_adj.row-1])

count_durg = np.zeros(n_drug, dtype=np.int)
for i in data.dp_edge_index[1, :]:
    count_durg[i] += 1
range_list = []
start = 0
end = 0
for i in count_durg:
    end += i
    range_list.append((start, end))
    start = end

data.dp_edge_index = torch.from_numpy(data.dp_edge_index + np.array([[0], [n_prot]]))
data.dp_range_list = range_list


data.d_norm = torch.ones(n_drug)
data.p_norm = torch.ones(n_prot+n_drug)
# data.x_norm = torch.sqrt(data.d_feat.sum(dim=1))
# data.d_feat.requires_grad = True


source_dim = n_prot_feat
embed_dim = 32
target_dim = 16


class HierEncoder(Module):
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


class NNDecoder(Module):
    def __init__(self, in_dim, num_uni_edge_type, l1_dim=8):
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


encoder = HierEncoder(source_dim, embed_dim, target_dim, n_prot, n_drug)
decoder = NNDecoder(target_dim, n_et_dd)
model = MyGAE(encoder, decoder)

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
# device_name = 'cpu'
print(device_name)
device = torch.device(device_name)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
data = data.to(device)

train_out = {}
test_out = {}



def train():

    model.train()

    optimizer.zero_grad()

    z = model.encoder(data.p_feat, data.dp_edge_index, data.dp_range_list, data.p_norm)

    pos_index = data.train_idx
    neg_index = negative_sampling(data.train_idx, n_drug).to(device)

    pos_score = model.decoder(z, pos_index, data.train_et)
    neg_score = model.decoder(z, neg_index, data.train_et)

    # pos_loss = F.binary_cross_entropy(pos_score, torch.ones(pos_score.shape[0]).cuda())
    # neg_loss = F.binary_cross_entropy(neg_score, torch.ones(neg_score.shape[0]).cuda())
    pos_loss = -torch.log(pos_score + EPS).mean()
    neg_loss = -torch.log(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss
    # loss = pos_loss


    loss.backward()
    optimizer.step()


    score = torch.cat([pos_score, neg_score])
    pos_target = torch.ones(pos_score.shape[0])
    neg_target = torch.zeros(neg_score.shape[0])
    target = torch.cat([pos_target, neg_target])
    auprc, auroc, ap = auprc_auroc_ap(target, score)
    # print(auprc, end='   ')

    print(epoch, ' ',
          'loss:', loss.tolist(), '  ',
          'auprc:', auprc, '  ',
          'auroc:', auroc, '  ',
          'ap:', ap)

    train_out[epoch] = [auprc, auroc, ap]

    return z, loss


test_neg_index = negative_sampling(data.test_idx, n_drug).to(device)


def test(z):
    model.eval()

    pos_score = model.decoder(z, data.test_idx, data.test_et)
    neg_score = model.decoder(z, test_neg_index, data.test_et)

    pos_target = torch.ones(pos_score.shape[0])
    neg_target = torch.zeros(neg_score.shape[0])

    score = torch.cat([pos_score, neg_score])
    target = torch.cat([pos_target, neg_target])

    auprc, auroc, ap = auprc_auroc_ap(target, score)

    return auprc, auroc, ap


EPOCH_NUM = 100
out_dir = '../out/pd-32-16-8-16-963/'

print('model training ...')
for epoch in range(EPOCH_NUM):
    time_begin = time.time()

    z, loss = train()

    auprc, auroc, ap = test(z)

    print(epoch, ' ',
          'loss:', loss.tolist(), '  ',
          'auprc:', auprc, '  ',
          'auroc:', auroc, '  ',
          'ap:', ap, '  ',
          'time:', time.time() - time_begin, '\n')

    # print(epoch, ' ',
    #       'auprc:', auprc)

    test_out[epoch] = [auprc, auroc, ap]

# save output to files
with open('../out/train_out.pkl', 'wb') as f:
    pickle.dump(train_out, f)

with open('../out/test_out.pkl', 'wb') as f:
    pickle.dump(test_out, f)

# save model state
filepath_state = out_dir + '100ep.pth'
torch.save(model.state_dict(), filepath_state)
# to restore
# model.load_state_dict(torch.load(filepath_state))
# model.eval()

# save whole model
filepath_model = out_dir + '100ep_model.pb'
torch.save(model, filepath_model)
# Then later:
# model = torch.load(filepath_model)


# ##################################
# training and testing figure
def dict_to_nparray(out_dict, epoch):
    out = np.zeros(shape=(3, epoch))
    for ep, [prc, roc, ap] in out_dict.items():
        out[0, ep] = prc
        out[1, ep] = roc
        out[2, ep] = ap
    return out


tr_out = dict_to_nparray(train_out, EPOCH_NUM)
te_out = dict_to_nparray(test_out, EPOCH_NUM)