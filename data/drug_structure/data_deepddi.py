import pickle
import os
import glob
import pandas as pd
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem

# load the map from drug id to index
with open('./data/index_map/drug-map.pkl', 'rb') as f:
    drug_map = pickle.load(f)

# inverse the index
drug_num = len(drug_map)
inversed_map = np.zeros(drug_num, dtype=np.int)
for k, v in drug_map.items():
    inversed_map[v] = k

# load the map from chem_id to DrugBank index
with open('./data/index_map/chem-map-db.pkl', 'rb') as f:
    chem_map_db = pickle.load(f)


def calculate_drug_similarity(drug_list, input_list,
                              output_file='./data/drug_structure/drug_similarity.csv'):
    drug_similarity_info = {}
    for each_drug_id1 in drug_list:
        drug_similarity_info[each_drug_id1] = {}
        drug1_mol = Chem.MolFromMolFile(each_drug_id1)
        drug1_mol = AllChem.AddHs(drug1_mol)
        for each_drug_id2 in input_list:
            drug2_mol = Chem.MolFromMolFile(each_drug_id2)
            drug2_mol = AllChem.AddHs(drug2_mol)
            fps = AllChem.GetMorganFingerprint(drug1_mol, 2)
            fps2 = AllChem.GetMorganFingerprint(drug2_mol, 2)
            score = DataStructs.DiceSimilarity(fps, fps2)
            drug_similarity_info[drugbank_id][input_drug_id] = score

    df = pd.DataFrame.from_dict(drug_similarity_info)
    df.to_csv(output_file)


inversed_db = []
input_db = []
count = 0
for i in inversed_map:
    if str(i) in chem_map_db:
        inversed_db.append(chem_map_db[str(i)])
        input_db.append(chem_map_db[str(i)])
        count += 1
    else:
        inversed_db.append('')
# only 345 chem comp

