from data.utils import load_data_torch, process_prot_edge
from src.utils import *
from src.layers import *
import pickle
import sys
import os
import time

with open('./data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

out_dir = './qu_out/ppm-ggm-nn'

MODEL = 'pp'

#########################################################################
# et_list = et_list[:10]       # remove this line for full dataset learning
#########################################################################

# et_list = et_list
feed_dict = load_data_torch("./data/", et_list, mono=True)
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
embed_dim = 64
target_dim = 32


encoder = HierEncoder(source_dim, embed_dim, target_dim, n_prot, n_drug)
# decoder = NNDecoder(target_dim, n_et_dd)
decoder = MultiInnerProductDecoder(target_dim, n_et_dd)
model = MyGAE(encoder, decoder)

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
# device_name = 'cpu'
print(device_name)
device = torch.device(device_name)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
data = data.to(device)


#########################################################################
# Load Trained Model
#########################################################################
filepath_model = os.path.join(out_dir, '100ep_model.pb')
model = torch.load(filepath_model)


#########################################################################
# Evaluation and Record
#########################################################################
def evaluate():
    model.eval()

    z = model.encoder(data.p_feat, data.dp_edge_index, data.dp_range_list, data.p_norm)
    
    pos_score = model.decoder(z, data.test_idx, data.test_et)
    
    return pos_score

print(data.test_idx.shape)
result = evaluate()

# with open('./qu_out/eva/ppm-ddm.pkl', 'wb') as f:
#     pickle.dump(result.tolist(), f)
print(result)
print(result.shape)

# with open('./qu_out/eva/tip-cat.pkl', 'rb') as f:
#     a = np.array([pickle.load(f)])