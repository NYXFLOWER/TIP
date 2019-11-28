from data.utils import load_data_torch, process_prot_edge
from src.utils import *
from src.layers import *
import pickle
import sys
import os
import time

with open('./TIP/data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

out_dir = './TIP/qu_out/tip-cat/'

EPOCH_NUM = 10

#########################################################################
et_list = et_list[:10]       # remove this line for full dataset learning
#########################################################################


feed_dict = load_data_torch("./TIP/data/", et_list, mono=True)

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



# ########################################
filepath_model = out_dir + '100ep_model.pb'
# torch.save(model, filepath_model)
# Then later:
model = torch.load(filepath_model)
print(model.encoder.embed)
print(model)


def evaluate():
    model.eval()

    z = model.encoder(data.d_feat, data.dd_train_idx, data.dd_train_et, data.dd_train_range, data.d_norm, data.p_feat, data.pp_train_indices, data.dp_edge_index, data.dp_range_list)
    
    pos_score = model.decoder(z, data.dd_test_idx, data.dd_test_et)
    return pos_score

result = evaluate()

with open('./TIP/qu_out/eva/tip-cat.pkl', 'wb') as f:
    pickle.dump(result.tolist(), f)