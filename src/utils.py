from sklearn import metrics
import scipy.sparse as sp
import numpy as np
import torch


def remove_bidirection(edge_index, edge_type):
    mask = edge_index[0] > edge_index[1]
    keep_set = mask.nonzero().view(-1)

    if edge_type is None:
        return edge_index[:, keep_set]
    else:
        return edge_index[:, keep_set], edge_type[keep_set]


def to_bidirection(edge_index, edge_type=None):
    tmp = edge_index.clone()
    tmp[0, :], tmp[1, :] = edge_index[1, :], edge_index[0, :]
    if edge_type is None:
        return torch.cat([edge_index, tmp], dim=1)
    else:
        return torch.cat([edge_index, tmp], dim=1), torch.cat([edge_type, edge_type])


def get_range_list(edge_list):
    tmp = []
    s = 0
    for i in edge_list:
        tmp.append((s, s + i.shape[1]))
        s += i.shape[1]
    return torch.tensor(tmp)


def process_edges(raw_edge_list, p=0.9):
    train_list = []
    test_list = []
    train_label_list = []
    test_label_list = []

    for i, idx in enumerate(raw_edge_list):
        train_mask = np.random.binomial(1, p, idx.shape[1])
        test_mask = 1 - train_mask
        train_set = train_mask.nonzero()[0]
        test_set = test_mask.nonzero()[0]

        train_list.append(idx[:, train_set])
        test_list.append(idx[:, test_set])

        train_label_list.append(torch.ones(2 * train_set.size, dtype=torch.long) * i)
        test_label_list.append(torch.ones(2 * test_set.size, dtype=torch.long) * i)

    train_list = [to_bidirection(idx) for idx in train_list]
    test_list = [to_bidirection(idx) for idx in test_list]

    train_range = get_range_list(train_list)
    test_range = get_range_list(test_list)

    train_edge_idx = torch.cat(train_list, dim=1)
    test_edge_idx = torch.cat(test_list, dim=1)

    train_et = torch.cat(train_label_list)
    test_et = torch.cat(test_label_list)

    return train_edge_idx, train_et, train_range, test_edge_idx, test_et, test_range


def sparse_id(n):
    idx = [[i for i in range(n)], [i for i in range(n)]]
    val = [1 for i in range(n)]
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(val)
    shape = (n, n)

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def dense_id(n):
    idx = [i for i in range(n)]
    val = [1 for i in range(n)]
    out = sp.coo_matrix((val, (idx, idx)), shape=(n, n), dtype=float)

    return torch.Tensor(out.todense())


def auprc_auroc_ap(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    auroc, ap = metrics.roc_auc_score(y, pred), metrics.average_precision_score(y, pred)
    y, xx, _ = metrics.precision_recall_curve(y, pred)
    auprc = metrics.auc(xx, y)

    return auprc, auroc, ap


def uniform(size, tensor):
    bound = 1.0 / np.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def dict_ep_to_nparray(out_dict, epoch):
    out = np.zeros(shape=(3, epoch))
    for ep, [prc, roc, ap] in out_dict.items():
        out[0, ep] = prc
        out[1, ep] = roc
        out[2, ep] = ap
    return out



