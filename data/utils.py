import numpy as np
import scipy.sparse as sp
import pickle
import torch
import torch.sparse as tsp
from torch_geometric.nn.models.autoencoder import negative_sampling
from src.utils import remove_bidirection, to_bidirection

torch.manual_seed(0)


def save_to_pkl(path, obj):
    with open(path, 'wb') as g:
        pickle.dump(obj, g)


def get_drug_index_from_text(code):
    return int(code.split('D')[-1])


def get_side_effect_index_from_text(code):
    return int(code.split('C')[-1])


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


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
        dd_adj_list.append(sp.triu(adj).tocsr())
        sum_adj += adj

    # ########################################
    # protein-protein
    # ########################################
    pp_adj = sp.load_npz(path + "sym_adj/protein-sparse-adj.npz")

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
    ind = torch.LongTensor([range(protein_num), range(protein_num)])
    val = torch.FloatTensor([1] * protein_num)
    protein_feat = torch.sparse.FloatTensor(ind, val,
                                            torch.Size([protein_num, protein_num]))

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

    ind = torch.LongTensor([row, col])
    val = torch.FloatTensor([1] * len(row))

    drug_feat = torch.sparse.FloatTensor(ind, val,
                                         torch.Size([drug_num, drug_num + mono_num]))

    # return a dict
    data = {'d_feat': drug_feat,
            'p_feat': protein_feat,
            'dd_adj_list': dd_adj_list,
            'dp_adj': dp_adj.tocoo(),
            'pp_adj': pp_adj.tocoo()}

    n_et = len(dd_et_list)

    num = [0]
    edge_index_list = []
    edge_type_list = []

    print(n_et, ' polypharmacy side effects')

    for i in range(n_et):
        # pos samples
        adj = dd_adj_list[i].tocoo()
        edge_index_list.append(torch.tensor([adj.row, adj.col], dtype=torch.long))
        edge_type_list.append(torch.tensor([i] * adj.nnz, dtype=torch.long))
        num.append(num[-1] + adj.nnz)

        # if i % 100 == 0:
        #     print(i)

    # data['dd_edge_index'] = t.cat(edge_index_list, 1)
    # data['dd_edge_type'] = t.cat(edge_type_list, 0)
    data['dd_edge_index'] = edge_index_list
    data['dd_edge_type'] = edge_type_list
    data['dd_edge_type_num'] = num
    data['dd_y_pos'] = torch.ones(num[-1])
    data['dd_y_neg'] = torch.zeros(num[-1])

    print('data has been loaded')

    return data


def cut_data(low, high, name, path='../data/', out_path='../out/'):
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
        if low < adj.nnz and adj.nnz < high:
            ind.append(i)

    with open(out_path + name + '.pkl', 'wb') as f:
        pickle.dump(ind, f)
    for i in ind:
        print(dd_adj_list[i].nnz)


def get_edge_list(low, name, source_path='./data/', out_path='./out/'):
    out = []
    for i in range(1317):
        adj = sp.load_npz(''.join([source_path, 'sym_adj/drug-sparse-adj/type_',
                                   str(i), '.npz']))
        if low < adj.nnz:
            out.append(i)

    with open(out_path + name + '.pkl', 'wb') as f:
        pickle.dump(out, f)

    return out


def process_prot_edge(pp_net):
    indices = torch.LongTensor(np.concatenate((pp_net.col.reshape(1, -1),
                                               pp_net.row.reshape(1, -1)),
                                              axis=0))
    indices = remove_bidirection(indices, None)
    n_edge = indices.shape[1]

    rd = np.random.binomial(1, 0.9, n_edge)
    train_mask = rd.nonzero()[0]
    test_mask = (1 - rd).nonzero()[0]

    train_indices = indices[:, train_mask]
    train_indices = to_bidirection(train_indices, None)

    test_indices = indices[:, test_mask]
    test_indices = to_bidirection(test_indices, None)

    return train_indices, test_indices
