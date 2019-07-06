
import numpy as np
import scipy.sparse as sp
import pickle


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
            dp_adj = sp.vstack([dp_adj[:ind, :], dp_adj[ind + 1:, :]]).tocsr()
            # remove from drug additional features
            if mono:
                drug_mono_adj = sp.vstack([drug_mono_adj[:ind, :], drug_mono_adj[ind + 1:, :]]).tocsr()

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
