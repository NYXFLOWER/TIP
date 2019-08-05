from data.utils import load_data_torch
from src.layers import negative_sampling, auprc_auroc_ap
import pickle
from torch.nn import Module
import torch
from src.layers import *
import sys
import time

sys.setrecursionlimit(8000)

with open('../data/training_samples_500.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

# et_list = et_list
feed_dict = load_data_torch("../data/", et_list, mono=True)

[n_drug, n_feat_d] = feed_dict['d_feat'].shape
n_et_dd = len(et_list)

data = Data.from_dict(feed_dict)

data.train_idx, data.train_et, data.train_range,data.test_idx, data.test_et, data.test_range = process_edges(data.dd_edge_index)


# TODO: add drug feature
data.d_feat = dense_id(n_drug)
n_feat_d = n_drug
data.x_norm = torch.ones(n_drug)
# data.x_norm = torch.sqrt(data.d_feat.sum(dim=1))
data.d_feat.requires_grad = True

n_base = 16

n_embed = 16
n_hid1 = 16
n_hid2 = 8


class Encoder(torch.nn.Module):

    def __init__(self, in_dim, num_et, num_base):
        super(Encoder, self).__init__()
        self.num_et = num_et

        self.embed = Param(torch.Tensor(in_dim, n_embed))
        self.rgcn1 = MyRGCNConv2(n_embed, n_hid1, num_et, num_base, after_relu=False)
        self.rgcn2 = MyRGCNConv2(n_hid1, n_hid2, num_et, num_base, after_relu=True)

        self.reset_paramters()

    def forward(self, x, edge_index, edge_type, range_list, x_norm):
        x = torch.matmul(x, self.embed)
        x = x / x_norm.view(-1, 1)
        x = self.rgcn1(x, edge_index, edge_type, range_list)
        x = F.relu(x, inplace=True)
        x = self.rgcn2(x, edge_index, edge_type, range_list)
        x = F.relu(x, inplace=True)
        return x

    def reset_paramters(self):
        self.embed.data.normal_()


class NNDecoder(Module):
    def __init__(self, in_dim, num_uni_edge_type, l1_dim=4):
        """ in_dim: the feat dim of a drug
            num_edge_type: num of dd edge type """

        super(NNDecoder, self).__init__()
        self.l1_dim = 4     # Decoder Lays' dim setting

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


encoder = Encoder(n_feat_d, n_et_dd, n_base)
decoder = NNDecoder(n_hid2, n_et_dd)
model = MyGAE(encoder, decoder)

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    z = model.encoder(data.d_feat, data.train_idx, data.train_et, data.train_range, data.x_norm)

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


print('model training ...')
for epoch in range(EPOCH_NUM):
    time_begin = time.time()

    z, loss = train()

    auprc, auroc, ap = test(z)

    print(epoch, ' ',
          'auprc:', auprc, '  ',
          'auroc:', auroc, '  ',
          'ap:', ap, '  ',
          'time:', time.time()-time_begin, '\n')

    # print(epoch, ' ',
    #       'auprc:', auprc)

    test_out[epoch] = [auprc, auroc, ap]


# save output to files
with open('../out/train_out.pkl', 'wb') as f:
    pickle.dump(train_out, f)

with open('../out/test_out.pkl', 'wb') as f:
    pickle.dump(test_out, f)
