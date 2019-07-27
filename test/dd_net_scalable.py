from data.utils import load_data_torch
from torch_geometric.data import Data
from torch_geometric.nn.models.autoencoder import GAE
import torch
from torch.nn.functional import binary_cross_entropy
import numpy as np
from sklearn import metrics
import pickle
import time
from tempfile import TemporaryFile
from src.layers import *

et_list = [20, 34, 38, 41, 42, 46, 55, 57, 89, 92, 99, 103, 105, 110, 125, 126, 129, 139, 147, 149, 152, 155, 157, 163,
           171, 174, 180, 190, 191, 198, 210, 216, 230, 240, 246, 251, 256, 260, 262, 264, 268, 273, 300, 306, 308, 309,
           321, 322, 324, 327, 336, 353, 354, 358, 359, 372, 373, 379, 382, 386, 388, 389, 390, 395, 397, 411, 412, 415,
           422, 425, 427, 428, 430, 432, 433, 435, 439, 447, 450, 451, 452, 453, 454, 455, 457, 459, 461, 462, 464, 466,
           468, 470, 471, 473, 481, 483, 484, 485, 490, 499, 502, 507, 511, 515, 517, 520, 525, 528, 529, 531, 535, 540,
           542, 552, 553, 559, 561, 563, 566, 568, 574, 579, 580, 581, 584, 586, 589, 591, 592, 594, 602, 605, 607, 618,
           620, 622, 627, 629, 634, 635, 636, 637, 639, 644, 645, 646, 651, 656, 657, 658, 662, 663, 664, 665, 666, 668,
           669, 671, 672, 673, 674, 680, 682, 684, 685, 687, 691, 694, 695, 696, 697, 699, 700, 705, 706, 710, 711, 713,
           714, 715, 717, 718, 720, 722, 725, 726, 727, 729, 730, 731, 735, 737, 739, 740, 742, 744, 748, 749, 752, 754,
           757, 759, 760, 761, 762, 763, 767, 768, 769, 770, 771, 772, 773, 776, 781, 782, 784, 787, 788, 793, 794, 795,
           797, 799, 801, 802, 803, 804, 805, 806, 809, 812, 813, 814, 815, 816, 818, 820, 821, 822, 825, 826, 827, 830,
           833, 834, 835, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 850, 851, 852, 853, 855, 856, 858, 859, 860,
           863, 865, 866, 867, 869, 875, 876, 878, 879, 881, 884, 885, 887, 889, 890, 891, 892, 893, 894, 895, 896, 897,
           898, 901, 903, 904, 905, 906, 907, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 921, 922, 923, 924,
           926, 930, 931, 933, 935, 939, 940, 941, 942, 943, 944, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956,
           958, 960, 961, 964, 965, 966, 967, 968, 969, 975, 976, 977, 978, 981, 982, 983, 984, 985, 989, 990, 992, 993,
           994, 995, 996, 997, 999, 1002, 1004, 1007, 1008, 1009, 1010, 1011, 1013, 1014, 1016, 1018, 1019, 1024, 1025,
           1026, 1027, 1029, 1031, 1032, 1033, 1034, 1036, 1037, 1039, 1048, 1049, 1050, 1051, 1054, 1055, 1060, 1062,
           1067, 1069, 1073, 1076, 1082, 1085, 1087, 1088, 1090, 1091, 1093, 1095, 1101, 1102, 1104, 1106, 1107, 1108,
           1112, 1116, 1118, 1123, 1126, 1128, 1133, 1137, 1138, 1145, 1148, 1152, 1153, 1171, 1181, 1205]

et_list = et_list
feed_dict = load_data_torch("../data/", et_list, mono=True)

[n_drug, n_feat_d] = feed_dict['d_feat'].shape
n_et_dd = len(et_list)

data = Data.from_dict(feed_dict)


data.train_idx, data.train_et, data.train_range,data.test_idx, data.test_et, data.test_range = process_edges(data.dd_edge_index)

# TODO: add drug feature
data.d_feat = sparse_id(n_drug)
n_feat_d = n_drug
data.x_norm = torch.ones(n_drug)
# data.x_norm = torch.sqrt(data.d_feat.sum(dim=1))

n_base = 16

n_embed = 64
n_hid1 = 32
n_hid2 = 16


class Encoder(torch.nn.Module):

    def __init__(self, in_dim, num_et, num_base):
        super(Encoder, self).__init__()
        self.num_et = num_et

        self.embed = Param(torch.Tensor(in_dim, n_embed))
        self.rgcn1 = MyRGCNConv2(n_embed, n_hid1, num_et, num_base, after_relu=False)
        self.rgcn2 = MyRGCNConv2(n_hid1, n_hid2, num_et, num_base, after_relu=True)

        self.reset_paramters()

    def forward(self, x, edge_index, edge_type, range_list, x_norm):
        x = torch.matmul(x, self.embed)
        x = x / x_norm.view(-1, 1)
        x = self.rgcn1(x, edge_index, edge_type, range_list)
        x = F.relu(x, inplace=True)
        x = self.rgcn2(x, edge_index, edge_type, range_list)
        x = F.relu(x, inplace=True)
        return x

    def reset_paramters(self):
        self.embed.data.normal_()


class MultiInnerProductDecoder(torch.nn.Module):
    def __init__(self, in_dim, num_et):
        super(MultiInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Param(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1/np.sqrt(self.in_dim))


encoder = Encoder(n_feat_d, n_et_dd, n_base)
decoder = MultiInnerProductDecoder(n_hid2, n_et_dd)
model = MyGAE(encoder, decoder)

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_name)
device = torch.device(device_name)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
data = data.to(device)


def train():
    model.train()

    optimizer.zero_grad()
    z = model.encoder(data.d_feat, data.train_idx, data.train_et, data.train_range, data.x_norm)

    pos_index = data.train_idx
    neg_index = negative_sampling(data.train_idx, n_drug).to(device)

    pos_score = model.decoder(z, pos_index, data.train_et)
    neg_score = model.decoder(z, neg_index, data.train_et)

    # pos_loss = F.binary_cross_entropy(pos_score, torch.ones(pos_score.shape[0]).cuda())
    # neg_loss = F.binary_cross_entropy(neg_score, torch.ones(neg_score.shape[0]).cuda())
    pos_loss = -torch.log(pos_score + EPS).mean()
    neg_loss = -torch.log(1 - neg_score + EPS).mean()
    loss = pos_loss + neg_loss
    # loss = pos_loss


    loss.backward()
    optimizer.step()


    score = torch.cat([pos_score, neg_score])
    pos_target = torch.ones(pos_score.shape[0])
    neg_target = torch.zeros(neg_score.shape[0])
    target = torch.cat([pos_target, neg_target])
    auprc, auroc, ap = auprc_auroc_ap(target, score)
    print(auprc, end='   ')

    return z, loss


test_neg_index = negative_sampling(data.test_idx, n_drug).to(device)


def test(z):
    model.eval()

    pos_score = model.decoder(z, data.test_idx, data.test_et)
    neg_score = model.decoder(z, test_neg_index, data.test_et)

    pos_target = torch.ones(pos_score.shape[0])
    neg_target = torch.zeros(neg_score.shape[0])

    score = torch.cat([pos_score, neg_score])
    target = torch.cat([pos_target, neg_target])

    auprc, auroc, ap = auprc_auroc_ap(target, score)

    return auprc, auroc, ap


EPOCH_NUM = 80


print('model training ...')
for epoch in range(EPOCH_NUM):
    z, loss = train()

    auprc, auroc, ap = test(z)

    # print(epoch, ' ',
    #       'auprc:', auprc, '  ',
    #       'auroc:', auroc, '  ',
    #       'ap:', ap)

    print(epoch, ' ',
          'auprc:', auprc)
