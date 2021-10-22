from data.utils import load_data_torch, process_prot_edge
from src.utils import *
import pickle

with open('./data/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)

out_file = './data/data_dict.pkl'

data = load_data_torch("./data/", et_list, mono=True)

# graph features
data['n_drug'] = data['d_feat'].shape[0]
data['n_prot'] = data['p_feat'].shape[0]
data['n_dd_et'] = len(et_list)

data['dd_train_idx'], data['dd_train_et'], data['dd_train_range'], data['dd_test_idx'], data['dd_test_et'], data['dd_test_range'] = process_edges(data['dd_edge_index'])
data['pp_train_indices'], data['pp_test_indices'] = process_prot_edge(data['pp_adj'])


# TODO: add drug feature
data['d_feat'] = sparse_id(data['n_drug'])
data['p_feat'] = sparse_id(data['n_prot'])
data['n_drug_feat'] = data['d_feat'].shape[1]
data['d_norm'] = torch.ones(data['n_drug_feat'])

# ###################################
# dp_edge_index and range index
# ###################################
data['dp_edge_index'] = np.array([data['dp_adj'].col-1, data['dp_adj'].row-1])

count_drug = np.zeros(data['n_drug'], dtype=np.int32)
for i in data['dp_edge_index'][1, :]:
    count_drug[i] += 1
range_list = []
start = 0
end = 0
for i in count_drug:
    end += i
    range_list.append((start, end))
    start = end

data['dp_edge_index'] = torch.from_numpy(data['dp_edge_index'] + np.array([[0], [data['n_prot']]]))
data['dp_range_list'] = torch.Tensor(range_list)

with open(out_file, 'wb') as f:
    pickle.dump(data, f)

print("Data has been prepared and is ready to use --> ./data/data_dict.pkl")