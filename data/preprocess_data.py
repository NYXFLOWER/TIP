import numpy as np
import scipy.sparse as sp
import pickle
import csv
from data.utils import get_side_effect_index_from_text, \
    get_drug_index_from_text


RAW_DATA_PATH = "../data/raw_data/"
WRITE_DATA_PATH = "../data/"

# index map initial
drug_map, drug_id = dict(), 0
protein_map, protein_id = dict(), 0
combo_map, combo_id = dict(), 0
mono_map, mono_id = dict(), 0

# ########################################
# drug-drug interaction
# ########################################
r, c = {}, {}
with open(RAW_DATA_PATH + "bio-decagon-combo.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)                # jump the title line

    for [i, j, k, _] in reader:
        ll = get_drug_index_from_text(i)
        m = get_drug_index_from_text(j)
        n = get_side_effect_index_from_text(k)

        if ll not in drug_map:
            drug_map[ll] = drug_id
            drug_id += 1
        if m not in drug_map:
            drug_map[m] = drug_id
            drug_id += 1
        if n not in combo_map:
            combo_map[n] = combo_id
            combo_id += 1

        ll, m, n = drug_map[ll], drug_map[m], combo_map[n]
        if n not in r:
            r[n], c[n] = [ll], [m]
        else:
            r[n].append(ll)
            c[n].append(m)

# build and save dd symmetric adjacency matrix
dd_sp_adj_list = []
for i in range(combo_id):
    adj = sp.coo_matrix((np.ones(len(r[i])), (r[i], c[i])),
                        shape=(drug_id, drug_id))
    sym_adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    dd_sp_adj_list.append(sym_adj)
    sp.save_npz(''.join([WRITE_DATA_PATH, 'sym_adj/drug-sparse-adj/type_', str(i), '.npz']), sym_adj)


# ########################################
# protein-protein interaction
# ########################################
r, c = [], []
with open(RAW_DATA_PATH + "bio-decagon-ppi.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)                # jump the title line
    for [i, j] in reader:
        i = int(i)
        j = int(j)

        if i not in protein_map:
            protein_map[i] = protein_id
            protein_id += 1
        if j not in protein_map:
            protein_map[j] = protein_id
            protein_id += 1

        i, j = protein_map[i], protein_map[j]
        r.append(i)
        c.append(j)

# build and save pp symmetric adjacency matrix
adj = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(protein_id, protein_id))
sym_adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
sp.save_npz(WRITE_DATA_PATH + "sym_adj/protein-sparse-adj.npz", sym_adj)


# ########################################
# drug-protein interaction
# ########################################
r, c = [], []                   # r - drug; c - protein
with open(RAW_DATA_PATH + "bio-decagon-targets.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)                # jump the title line

    for [i, j] in reader:
        m = get_drug_index_from_text(i)
        n = int(j)

        if m not in drug_map or n not in protein_map:
            continue

        m, n = drug_map[m], protein_map[n]
        r.append(m)
        c.append(n)

# build and save drug-protein adjacency matrix
adj = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(drug_id, protein_id))
sp.save_npz(WRITE_DATA_PATH + "sym_adj/drug-protein-sparse-adj.npz", adj)


# #####################################################
# drug additional feature - single drug side effect
# #####################################################
r, c = [], []                   # r - drug; c - single drug side effect
with open(RAW_DATA_PATH + "bio-decagon-mono.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)                # jump the title line

    for [i, j, _] in reader:
        ll = get_drug_index_from_text(i)
        m = get_side_effect_index_from_text(j)

        if ll not in drug_map:
            continue
        if m not in mono_map:
            mono_map[m] = mono_id
            mono_id += 1

        ll, m = drug_map[ll], mono_map[m]
        r.append(ll)
        c.append(m)

# build and save drug additional feature matrix
adj = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(drug_id, mono_id))
sp.save_npz(WRITE_DATA_PATH + "node_feature/drug-mono-feature.npz", adj)


# #####################################################
# save graph info
# #####################################################
def save_to_pkl(path, obj):
    with open(path, 'wb') as ff:
        pickle.dump(obj, ff)


save_to_pkl(WRITE_DATA_PATH+"index_map/drug-map.pkl", drug_map)
save_to_pkl(WRITE_DATA_PATH+"index_map/protein-map.pkl", protein_map)
save_to_pkl(WRITE_DATA_PATH+"index_map/combo_map.pkl", combo_map)
save_to_pkl(WRITE_DATA_PATH+"index_map/mono_map.pkl", mono_map)

save_to_pkl(WRITE_DATA_PATH+"graph_info.pkl", (drug_id, protein_id, combo_id, mono_id))


# #####################################################
# chem id to drugbank id
# #####################################################
hhh = []
with open("./data/index_map/drug links.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for drug_info in reader:
        # if drug_info[6] and drug_info[-7]:
        #     hhh.append([drug_info[0], drug_info[6], drug_info[-7]])
        if drug_info[6]:
            hhh.append([drug_info[0], drug_info[6]])

chem_map_db = {}
# db_map_smile = {}
# for db_id, smiles, chem_id in hhh:
for db_id, chem_id in hhh:
    chem_map_db[chem_id] = db_id
    # db_map_smile[db_id] = smiles

save_to_pkl('./data/index_map/chem-map-db.pkl', chem_map_db)
# save_to_pkl('./data/index_map/db_map_smile.pkl', db_map_smile)
