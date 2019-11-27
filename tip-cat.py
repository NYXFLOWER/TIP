from data.utils import load_data_torch, process_prot_edge
from src.utils import *
from src.layers import *
import pickle
import sys
import os
import time
# import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

with open('./data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)


#########################################################################
et_list = et_list[:10]       # remove this line for full dataset learning
#########################################################################


feed_dict = load_data_torch("./data/", et_list, mono=True)

data = Data.from_dict(feed_dict)
data.n_drug = data.d_feat.shape[0]
data.n_prot = data.p_feat.shape[0]
data.n_dd_et = len(et_list)

data.dd_train_idx, data.dd_train_et, data.dd_train_range, data.dd_test_idx, data.dd_test_et, data.dd_test_range = process_edges(data.dd_edge_index)
data.pp_train_indices, data.pp_test_indices = process_prot_edge(data.pp_adj)


# TODO: add drug feature
data.d_feat = sparse_id(data.n_drug)
data.p_feat = sparse_id(data.n_prot)
data.n_drug_feat = data.d_feat.shape[1]
data.d_norm = torch.ones(data.n_drug_feat)

# ###################################
# dp_edge_index and range index
# ###################################
data.dp_edge_index = np.array([data.dp_adj.col-1, data.dp_adj.row-1])

count_drug = np.zeros(data.n_drug, dtype=np.int)
for i in data.dp_edge_index[1, :]:
    count_drug[i] += 1
range_list = []
start = 0
end = 0
for i in count_drug:
    end += i
    range_list.append((start, end))
    start = end

data.dp_edge_index = torch.from_numpy(data.dp_edge_index + np.array([[0], [data.n_prot]]))
data.dp_range_list = torch.Tensor(range_list)

# data.d_feat.requires_grad = True
# data.p_feat.requires_grad = True

out_dir = './out_new/tip-cat/'


class PPEncoder(torch.nn.Module):

    def __init__(self, in_dim, hid1=32, hid2=16):
        super(PPEncoder, self).__init__()
        self.out_dim = hid2

        self.conv1 = GCNConv(in_dim, hid1, cached=True)
        self.conv2 = GCNConv(hid1, hid2, cached=True)

        # self.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x, inplace=True)
        x = self.conv2(x, edge_index)
        return x

    # def reset_parameters(self):
    #     self.embed.data.normal_()


class FMEncoder(torch.nn.Module):

    def __init__(self, device, in_dim_drug, num_dd_et, in_dim_prot,
                 uni_num_prot, uni_num_drug, prot_drug_dim=16,
                 num_base=16, n_embed=48, n_hid1=32, n_hid2=16, mod='cat'):
        '''
        :param device:
        :param in_dim_drug:
        :param num_dd_et:
        :param in_dim_prot:
        :param uni_num_prot:
        :param uni_num_drug:
        :param prot_drug_dim:
        :param num_base:
        :param n_embed:
        :param n_hid1:
        :param n_hid2:
        :param mod: 'cat', 'ave'
        '''
    # def __init__(self, device, in_dim_drug, num_dd_et, in_dim_prot,
    #              uni_num_prot, uni_num_drug, prot_drug_dim=9,
    #              num_base=6, n_embed=4, n_hid1=2, n_hid2=2):
        super(FMEncoder, self).__init__()
        self.num_et = num_dd_et
        self.out_dim = n_hid2
        self.uni_num_drug = uni_num_drug
        self.uni_num_prot = uni_num_prot

        # on pp-net
        self.pp_encoder = PPEncoder(in_dim_prot)

        # feat: drug index
        self.embed = Param(torch.Tensor(in_dim_drug, n_embed))

        # on pd-net
        self.hgcn = MyHierarchyConv(self.pp_encoder.out_dim, prot_drug_dim, uni_num_prot, uni_num_drug)
        self.hdrug = torch.zeros((self.uni_num_drug, self.pp_encoder.out_dim)).to(device)

        # on dd-net
        self.rgcn1 = MyRGCNConv2(n_embed+self.hgcn.out_dim, n_hid1, num_dd_et, num_base, after_relu=False)
        self.rgcn2 = MyRGCNConv2(n_hid1, n_hid2, num_dd_et, num_base, after_relu=True)

        self.reset_parameters()

    def forward(self, x_drug, dd_edge_index, dd_edge_type, dd_range_list, d_norm,
                x_prot, pp_edge_index, dp_edge_index, dp_range_list):
        # pp-net
        x_prot = self.pp_encoder(x_prot, pp_edge_index)
        # x_prot = checkpoint(self.pp_encoder, x_prot, pp_edge_index)
        # pd-net
        x_prot = torch.cat((x_prot, self.hdrug))
        x_prot = self.hgcn(x_prot, dp_edge_index, dp_range_list)
        # x_prot = checkpoint(self.hgcn, torch.cat((x_prot, self.hdrug)), dp_edge_index, dp_range_list)

        # d-embed
        x_drug = torch.matmul(x_drug, self.embed)
        # x_drug = checkpoint(torch.matmul, x_drug, self.embed)
        x_drug = x_drug / d_norm.view(-1, 1)
        x_drug = torch.cat((x_drug, x_prot), dim=1)

        # dd-net
        # x = self.rgcn1(x, edge_index, edge_type, range_list)
        x_drug = checkpoint(self.rgcn1, x_drug, dd_edge_index, dd_edge_type, dd_range_list)

        x_drug = F.relu(x_drug, inplace=True)
        x_drug = self.rgcn2(x_drug, dd_edge_index, dd_edge_type, dd_range_list)
        # x_drug = checkpoint(self.rgcn2, x_drug, dd_edge_index, dd_edge_type, dd_range_list)
        return x_drug

    def reset_parameters(self):
        self.embed.data.normal_()


class MultiInnerProductDecoder(torch.nn.Module):
    def __init__(self, in_dim, num_et):
        super(MultiInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Param(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1/np.sqrt(self.in_dim))


device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_name)
device = torch.device(device_name)

encoder = FMEncoder(device, data.n_drug_feat, data.n_dd_et, data.n_prot, data.n_prot, data.n_drug)
decoder = MultiInnerProductDecoder(encoder.out_dim, data.n_dd_et)
model = MyGAE(encoder, decoder)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
data = data.to(device)

train_record = {}
test_record = {}
train_out = {}
test_out = {}

##################################################
# @profile        # remove this for training on CPU
##################################################
def train():
    model.train()

    optimizer.zero_grad()
    z = model.encoder(data.d_feat, data.dd_train_idx, data.dd_train_et, data.dd_train_range, data.d_norm, data.p_feat, data.pp_train_indices, data.dp_edge_index, data.dp_range_list)

    pos_index = data.dd_train_idx
    neg_index = typed_negative_sampling(data.dd_train_idx, data.n_drug, data.dd_train_range).to(device)

    pos_score = checkpoint(model.decoder, z, pos_index, data.dd_train_et)
    neg_score = checkpoint(model.decoder, z, neg_index, data.dd_train_et)

    # pos_loss = F.binary_cross_entropy(pos_score, torch.ones(pos_score.shape[0]).cuda())
    # neg_loss = F.binary_cross_entropy(neg_score, torch.ones(neg_score.shape[0]).cuda())
    pos_loss = -torch.log(pos_score + EPS).mean()
    neg_loss = -torch.log(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss
    # loss = pos_loss

    loss.backward()
    optimizer.step()

    record = np.zeros((3, data.n_dd_et))  # auprc, auroc, ap
    for i in range(data.dd_train_range.shape[0]):
        [start, end] = data.dd_train_range[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target,
                                                                  score)

    train_record[epoch] = record
    [auprc, auroc, ap] = record.sum(axis=1) / data.n_dd_et
    train_out[epoch] = [auprc, auroc, ap]

    print('{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}'
          .format(epoch, loss.tolist(), auprc, auroc, ap))

    return z, loss


test_neg_index = typed_negative_sampling(data.dd_test_idx, data.n_drug, data.dd_test_range).to(device)


def test(z):
    model.eval()

    record = np.zeros((3, data.n_dd_et))     # auprc, auroc, ap

    pos_score = model.decoder(z, data.dd_test_idx, data.dd_test_et)
    neg_score = model.decoder(z, test_neg_index, data.dd_test_et)

    for i in range(data.dd_test_range.shape[0]):
        [start, end] = data.dd_test_range[i]
        p_s = pos_score[start: end]
        n_s = neg_score[start: end]

        pos_target = torch.ones(p_s.shape[0])
        neg_target = torch.zeros(n_s.shape[0])

        score = torch.cat([p_s, n_s])
        target = torch.cat([pos_target, neg_target])

        record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)

    return record


EPOCH_NUM = 100

print('model training ...')

for epoch in range(EPOCH_NUM):
    time_begin = time.time()

    z, loss = train()

    record_te = test(z)
    [auprc, auroc, ap] = record_te.sum(axis=1) / data.n_dd_et

    print('{:3d}   loss:{:0.4f}   auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}    time:{:0.1f}\n'
          .format(epoch, loss.tolist(), auprc, auroc, ap, (time.time() - time_begin)))

    test_record[epoch] = record_te
    test_out[epoch] = [auprc, auroc, ap]

#
# save output to files
with open(out_dir + 'train_out.pkl', 'wb') as f:
    pickle.dump(train_out, f)

with open(out_dir + 'test_out.pkl', 'wb') as f:
    pickle.dump(test_out, f)

with open(out_dir + 'train_record.pkl', 'wb') as f:
    pickle.dump(train_record, f)

with open(out_dir + 'test_record.pkl', 'wb') as f:
    pickle.dump(test_record, f)

# save model state
filepath_state = out_dir + '100ep.pth'
torch.save(model.state_dict(), filepath_state)
# to restore
# model.load_state_dict(torch.load(filepath_state))
# model.eval()

# save whole model
# out_dir = './out/fm-(32-16)-(16-16-48-32-16)/'
filepath_model = out_dir + '100ep_model.pb'
torch.save(model, filepath_model)
# Then later:
# model = torch.load(filepath_model)


# ##################################
# training and testing figure
# tr_out = dict_ep_to_nparray(train_out, EPOCH_NUM)
# te_out = dict_ep_to_nparray(test_out, EPOCH_NUM)
#
# plt.figure()
# x = np.array(range(0, EPOCH_NUM, 5), dtype=int) + 1
# maxmum = np.zeros(EPOCH_NUM) + te_out[0, :].max()
# plt.plot(x, tr_out[0, :], label='train_prc')
# plt.plot(x, te_out[0, :], label='test_prc')
# plt.plot(x, maxmum, linestyle="-.")
# plt.title('AUPRC scores - FM')
# plt.grid()
# plt.legend()
# plt.savefig(out_dir + 'prc.png')
# plt.show()
