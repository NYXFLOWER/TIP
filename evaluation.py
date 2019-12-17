from data.utils import load_data_torch, process_prot_edge
from src.utils import *
from src.layers import *
import pickle
import sys
import os
import time

with open('./data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

out_dir = './qu_out/tip-cat'

EPOCH_NUM = 10
MODEL = 'tip-cat'

#########################################################################
# et_list = et_list[:10]       # remove this line for full dataset learning
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

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_name)
device = torch.device(device_name)

# encoder
if MODEL == 'tip-cat':
    encoder = FMEncoderCat(device, data.n_drug_feat, data.n_dd_et, data.n_prot,
                           data.n_prot, data.n_drug)
elif MODEL == 'tip-add':
    encoder = FMEncoder(device, data.n_drug_feat, data.n_dd_et, data.n_prot,
                           data.n_prot, data.n_drug)

decoder = MultiInnerProductDecoder(encoder.out_dim, data.n_dd_et)
model = MyGAE(encoder, decoder)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
data = data.to(device)


#########################################################################
# Load Trained Model
#########################################################################
filepath_model = os.path.join(out_dir, '100ep_model.pb')
model = torch.load(filepath_model)


#########################################################################
#
#########################################################################
def evaluate():
    model.eval()

    z = model.encoder(data.d_feat, data.dd_train_idx, data.dd_train_et, data.dd_train_range, data.d_norm, data.p_feat, data.pp_train_indices, data.dp_edge_index, data.dp_range_list)
    
    pos_score = model.decoder(z, data.dd_test_idx, data.dd_test_et)
    return pos_score


result = evaluate()

with open('./qu_out/eva/{}.pkl'.format(MODEL), 'wb') as f:
    pickle.dump(result.tolist(), f)

# import numpy as np 
# import pickle
# with open('./qu_out/eva/tip-cat.pkl', 'rb') as f:
#     a = np.array([pickle.load(f)])
# print(a)