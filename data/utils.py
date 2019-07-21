import numpy as np
import scipy.sparse as sp
import pickle
import torch as t
import torch.sparse as tsp
from torch_geometric.nn.models.autoencoder import negative_sampling

t.manual_seed(0)


def save_to_pkl(path, obj):
    with open(path, 'wb') as g:
        pickle.dump(obj, g)


def get_drug_index_from_text(code):
    return int(code.split('D')[-1])


def get_side_effect_index_from_text(code):
    return int(code.split('C')[-1])


def load_data_torch(path, dd_et_list, mono=True):
    """
    :param path: WRITE_DATA_PATH in preprocess_data.py
    :param dd_et_list: a list of int - drug indices
    :param mono: if consider single drug side effects as drug features
    :return: a dict contain: dd-adj list, pp-adj, dp-adj and the feature matrix of drug and protein
    """

    # path = './data/'
    # dd_et_list = [0, 1, 2, 3]

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
        adj = sp.load_npz(
            ''.join([path, 'sym_adj/drug-sparse-adj/type_', str(i), '.npz']))

        # dd_adj_list.append(adj)
        dd_adj_list.append(sp.triu(adj))
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
        drug_mono_adj = sp.load_npz(
            path + "node_feature/drug-mono-feature.npz").tocsr()

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
                dd_adj_list[i] = sp.vstack([dd_adj_list[i][:ind, :],
                                            dd_adj_list[i][ind + 1:,
                                            :]]).tocsr()
                dd_adj_list[i] = sp.hstack([dd_adj_list[i][:, :ind],
                                            dd_adj_list[i][:,
                                            ind + 1:]]).tocsr()
            # remove from d-p adj
            dp_adj = sp.vstack([dp_adj[:ind, :], dp_adj[ind + 1:, :]])
            # remove from drug additional features
            if mono:
                drug_mono_adj = sp.vstack(
                    [drug_mono_adj[:ind, :], drug_mono_adj[ind + 1:, :]])
    print('remove finished')
    # ########################################
    # protein feature matrix
    # ########################################
    # protein_feat = sp.identity(protein_num)
    ind = t.LongTensor([range(protein_num), range(protein_num)])
    val = t.FloatTensor([1] * protein_num)
    protein_feat = t.sparse.FloatTensor(ind, val,
                                        t.Size([protein_num, protein_num]))

    # ########################################
    # drug feature matrix
    # ########################################
    row = np.array(range(drug_num), dtype=np.long)
    col = np.array(range(drug_num), dtype=np.long)
    # drug_feat = sp.identity(drug_num - isolated_num)
    if mono:
        # drug_feat = sp.hstack([drug_feat, drug_mono_adj])
        adj = drug_mono_adj.tocoo()
        row = np.append(row, adj.row)
        col = np.append(col, adj.col + drug_num)
    else:
        mono_num = 0

    ind = t.LongTensor([row, col])
    val = t.FloatTensor([1] * len(row))

    drug_feat = t.sparse.FloatTensor(ind, val, t.Size([drug_num, drug_num
                                                       + mono_num]))

    # return a dict
    data = {'d_feat': drug_feat,
            'p_feat': protein_feat,
            'dd_adj_list': dd_adj_list,
            'dp_adj': dp_adj,
            'pp_adj': pp_adj}

    n_et = len(dd_et_list)

    num = [0]
    edge_index_list = []
    edge_type_list = []

    print(n_et, ' polypharmacy side effects')

    for i in range(n_et):
        # pos samples
        adj = dd_adj_list[i].tocoo()
        edge_index_list.append(t.tensor([adj.row, adj.col], dtype=t.long))
        edge_type_list.append(t.tensor([i] * adj.nnz, dtype=t.long))
        num.append(num[-1] + adj.nnz)

        # if i % 100 == 0:
        #     print(i)

    # data['dd_edge_index'] = t.cat(edge_index_list, 1)
    # data['dd_edge_type'] = t.cat(edge_type_list, 0)
    data['dd_edge_index'] = edge_index_list
    data['dd_edge_type'] = edge_type_list
    data['dd_edge_type_num'] = num
    data['dd_y_pos'] = t.ones(num[-1])
    data['dd_y_neg'] = t.zeros(num[-1])

    print('data has been loaded')

    return data


def cut_data(low, high, name, path='/Users/nyxfer/Docu/FM-PSEP/data/'):
    dd_et_list = list(range(1317))

    with open(path + 'graph_info.pkl', 'rb') as f:
        drug_num, _, _, _ = pickle.load(f)

    dd_adj_list = []
    sum_adj = sp.csr_matrix((drug_num, drug_num))
    for i in dd_et_list:
        adj = sp.load_npz(
            ''.join([path, 'sym_adj/drug-sparse-adj/type_', str(i), '.npz']))
        dd_adj_list.append(adj)
        sum_adj += adj

    ind = []
    for i in range(1317):
        adj = dd_adj_list[i]
        if low < adj.nnz < high:
            ind.append(i)

    with open('./data/' + name + '.pkl', 'wb') as f:
        pickle.dump(ind, f)
    # for i in ind:
    #     print(dd_adj_list[i].nnz)
