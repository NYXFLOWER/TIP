from data.utils import load_data_torch
from src.layers import negative_sampling, auprc_auroc_ap
import pickle
from torch.nn import Module
import torch
from src.layers import *

# Decoder Lays' dim setting
l1_dim = 64

with open('./data/1k-5k.pkl', 'rb') as f:       # 425 dd edge types
    et_list = pickle.load(f)


class NNDecoder(Module):
    def __init__(self, in_dim, num_et, ):
        super(NNDecoder, self).__init__()
        self.num_et = num_et

        # parameters
        self.w1 = Param(torch.Tensor(in_dim, l1_dim))
        self.w2 = Param(torch.Tensor(l1_dim, num_et))

        self.reset_parameters()

    def forward(self, input):
        

    def reset_parameters(self):
        self.w1.data.normal_()
        self.w2.data.normal_()

