import numpy as np
import scipy.sparse as sp
import pickle
import torch as t
from torch_geometric.nn.models.autoencoder import negative_sampling

t.manual_seed(0)


def get_drug_index_from_text(code):
    return int(code.split('D')[-1])


def get_side_effect_index_from_text(code):
    return int(code.split('C')[-1])


def load_data(path, dd_et_list, mono=True):
    """
    :param path: WRITE_DATA_PATH in preprocess_data.py
    :param dd_et_list: a list of int - drug indices
    :param mono: if consider single drug side effects as drug features
    :return: a dict contain: dd-adj list, pp-adj, dp-adj and the feature matrix of drug and protein
    """
    print("loading data")
    # load graph info
    with open(path + 'graph_info.pkl', 'rb') as f:
        drug_num, protein_num, combo_num, mono_num = pickle.load(f)

    # ########################################
    # drug-drug
    # ########################################
    dd_adj_list = []
    sum_adj = sp.csr_matrix((drug_num, drug_num))
    for i in dd_et_list:
        adj = sp.load_npz(''.join([path, 'sym_adj/drug-sparse-adj/type_', str(i), '.npz']))
        dd_adj_list.append(adj)
        sum_adj += adj

    # ########################################
    # protein-protein
    # ########################################
    pp_adj = sp.load_npz(path + "sym_adj/protein-sparse-adj.npz").tocsr()

    # ########################################
    # drug-protein
    # ########################################
    dp_adj = sp.load_npz(path + "sym_adj/drug-protein-sparse-adj.npz").tocsr()

    # ########################################
    # drug additional feature
    # ########################################
    if mono:
        drug_mono_adj = sp.load_npz(path + "node_feature/drug-mono-feature.npz").tocsr()

    # ########################################
    # remove isolated drugs
    # ########################################
    drug_degree = sum_adj.sum(axis=1)
    isolated_drug = np.where(drug_degree == 0)[0].tolist()
    isolated_num = len(isolated_drug)
    print("remove ", isolated_num, " isolated drugs: ", isolated_drug)
    if len(isolated_drug) != 0:
        while len(isolated_drug) != 0:
            ind = isolated_drug.pop()
            # remove from d-d adj
            for i in range(len(dd_adj_list)):
                dd_adj_list[i] = sp.vstack([dd_adj_list[i][:ind, :], dd_adj_list[i][ind + 1:, :]]).tocsr()
                dd_adj_list[i] = sp.hstack([dd_adj_list[i][:, :ind], dd_adj_list[i][:, ind + 1:]]).tocsr()
            # remove from d-p adj
            dp_adj = sp.vstack([dp_adj[:ind, :], dp_adj[ind + 1:, :]])
            # remove from drug additional features
            if mono:
                drug_mono_adj = sp.vstack([drug_mono_adj[:ind, :], drug_mono_adj[ind + 1:, :]])
    print('remove finished')
    # ########################################
    # protein feature matrix
    # ########################################
    protein_feat = sp.identity(protein_num)

    # ########################################
    # drug feature matrix
    # ########################################
    drug_feat = sp.identity(drug_num - isolated_num)
    if mono:
        drug_feat = sp.hstack([drug_feat, drug_mono_adj])

    # return a dict
    data = {'d_feat': drug_feat,
            'p_feat': protein_feat,
            'dd_adj_list': dd_adj_list,
            'dp_adj': dp_adj,
            'pp_adj': pp_adj}
    return data


def load_data_torch(path, dd_et_list, mono=True):
    data = load_data(path, dd_et_list, mono=mono)
    data['d_feat'] = t.tensor(data['d_feat'].toarray(), dtype=t.float32)

    n_et = len(dd_et_list)
    n_drug = data['d_feat'].shape[0]
    adj_list = data['dd_adj_list']

    num = [0]
    edge_index_list = []
    edge_type_list = []

    print(n_et, ' polypharmacy side effects')

    for i in range(n_et):
        # pos samples
        adj = adj_list[i].tocoo()
        edge_index_list.append(t.tensor([adj.row, adj.col], dtype=t.long))
        edge_type_list.append(t.tensor([i] * adj.nnz, dtype=t.long))
        num.append(num[-1] + adj.nnz)

        # if i % 100 == 0:
        #     print(i)

    data['dd_edge_index'] = t.cat(edge_index_list, 1)
    data['dd_edge_type'] = t.cat(edge_type_list, 0)
    data['dd_edge_type_num'] = num
    data['dd_y_pos'] = t.ones(num[-1])
    data['dd_y_neg'] = t.zeros(num[-1])

    print('data has been loaded')

    return data


# with open("/Users/nyxfer/Docu/FM-PSEP/data/training_samples_500.pkl", "rb") as f:
#     et_list = pickle.load(f)
# data = load_data_torch("/Users/nyxfer/Docu/FM-PSEP/data/", et_list, mono=True)





