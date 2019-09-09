from data.utils import load_data_torch
import numpy as np
import pickle
import csv

# drug id - index
with open('../data/index_map/drug-map.pkl', 'rb') as f:
    drug_map = pickle.load(f)
inv_drug_map = {v: k for k, v in drug_map.items()}

# combo id - index
with open('../data/index_map/combo_map.pkl', 'rb') as f:
    combo_map = pickle.load(f)
inv_combo_map = {v: k for k, v in combo_map.items()}

# selected-drug idx - drug idx
with open('../out/decagon_et.pkl', 'rb') as f:   # the whole dataset
    et_list = pickle.load(f)
inv_et_list = {et_list[i]: i for i in range(len(et_list))}

feed_dict = load_data_torch("../data/", et_list, mono=False)

# ######################################################
# generate polypharmacy side effect id - name map
# combo_name_map = {}
# with open('../data/index_map/bio-decagon-combo.csv', 'r') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for _, _, id, name in reader:
#         id = int(id.split('C')[-1])
#         combo_name_map[id] = name
#
# # save map
# with open('../data/index_map/combo-name-map.pkl', 'wb') as f:
#     pickle.dump(combo_name_map, f)

# use map
with open('../data/index_map/combo-name-map.pkl', 'rb') as f:
    combo_name_map = pickle.load(f)

# ######################################################
# side effect name - original index reported in decagon
decagon_best_name = ["Mumps", "Carbuncle", "Coccydynia", "Tympanic membrane perfor", "Dyshidrosis", "Spondylosis", "Schizoaffective disorder", "Breast dysplasia", "Ganglion", "Uterine polyp"]
decagon_worst_name = ["Bleeding", "Body temperature increased",  "Emesis", "Renal disorder", "Leucopenia", "Diarrhea", "Icterus", "Nausea", "Itch", "Anaemia"]
decagon_best_org_id = [26780, 7078, 9193, 206504, 32633, 38019, 36337, 16034, 1258666, 156369]
decagon_worst_org_id = [19080, 15967, 42963, 22658, 23530, 11991, 22346, 27497, 33774, 2871]

# get index
decagon_best_idx = [inv_et_list[combo_map[i]] for i in decagon_best_org_id]
decagon_worst_idx = [inv_et_list[combo_map[i]] for i in decagon_worst_org_id]

# ######################################################
# Evaluation
name = 'RGCN-DistMult on d-net'
with open('../out/dd-rgcn-dist(16-64-32-16)/test_record.pkl', 'rb') as f:
    dist_record = pickle.load(f)
auprc = np.array(dist_record[len(dist_record)-1])[0, :]
sorted_idx = np.argsort(auprc, kind='quicksort')

print(' {:37s}   {:6s}| {:45s}  {:6s}'.format('The Highest AUPRC Score', '  Edge', 'The Lowest AUPRC Score', '   Edge'))
for i in range(20):
    print(' {:30s} {:7.4f}  {:6d}| {:38s} {:7.4f}  {:6d}'.format(
        combo_name_map[inv_combo_map[et_list[sorted_idx[-(i+1)]]]], auprc[sorted_idx[-(i+1)]], feed_dict['dd_adj_list'][-(i+1)].nnz,
        combo_name_map[inv_combo_map[et_list[sorted_idx[i]]]], auprc[sorted_idx[i]], feed_dict['dd_adj_list'][i].nnz))

decag_best_in_us = [962 - np.where(sorted_idx == i)[0] for i in decagon_best_idx]
decag_worst_in_us = [np.where(sorted_idx == i)[0] for i in decagon_worst_idx]

