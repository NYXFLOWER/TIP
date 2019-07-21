from data.utils import load_data
from matplotlib import pyplot as plt
from src.layers import *

path = r"../data/"
pp_net = load_data(path, [])['pp_adj'].tocoo()
indices = torch.LongTensor(np.concatenate((pp_net.col.reshape(1, -1),
                                       pp_net.row.reshape(1, -1)), axis=0))
indices = remove_bidirection(indices, None)

n_node = pp_net.shape[0]
n_edge = indices.shape[1]

rd = np.random.binomial(1, 0.9, n_edge)
train_mask = rd.nonzero()[0]
test_mask = (1 - rd).nonzero()[0]

train_indices = indices[:, train_mask]
train_indices = to_bidirection(train_indices, None)

test_indices = indices[:, test_mask]
test_indices = to_bidirection(test_indices, None)

train_n_edge = train_indices.shape[1]
test_n_edge = test_indices.shape[1]

hid1 = 32
hid2 = 16

x = sparse_id(n_node)


class PP_Encoder(torch.nn.Module):
    def __init__(self):
        super(PP_Encoder, self).__init__()
        self.conv1 = GCNConv(n_node, hid1, cached=True)
        self.conv2 = GCNConv(hid1, hid2, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.normalize(x, dim=1)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


device = torch.device('cuda')
model = GAE(PP_Encoder()).to(device)
train_indices = train_indices.to(device)
test_indices = test_indices.to(device)
x = x.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=10)
neg_test_indices = negative_sampling(test_indices, n_node).to(device)


for i in range(80):
    # train part
    model.train()
    optimizer.zero_grad()

    z = model.encoder(x, train_indices)
    pos_indices = train_indices
    neg_indices = negative_sampling(train_indices, n_node)

    pos_score = model.decoder(z, pos_indices)
    neg_score = model.decoder(z, neg_indices)

    pos_loss = F.binary_cross_entropy(pos_score, torch.ones(train_n_edge).cuda())
    neg_loss = F.binary_cross_entropy(neg_score, torch.zeros(train_n_edge).cuda())
    loss = neg_loss + pos_loss

    loss.backward()
    optimizer.step()

    # test
    model.eval()

    pos_score = model.decoder(z, test_indices)
    neg_score = model.decoder(z, neg_test_indices)

    score = torch.cat([pos_score, neg_score])
    target = torch.cat([torch.ones(test_n_edge), torch.zeros(test_n_edge)])

    y, pred = target.detach().cpu().numpy(), score.detach().cpu().numpy()

    auroc, ap = metrics.roc_auc_score(y, pred), metrics.average_precision_score(y, pred)
    y, xx, _ = metrics.ranking.precision_recall_curve(y, pred)
    auprc = metrics.ranking.auc(xx, y)

    pos_acc = (pos_score > 0.5).sum().to(torch.float) / float(n_edge)
    neg_acc = (neg_score < 0.5).sum().to(torch.float) / float(n_edge)


    # print('pos_loss: ', pos_loss.tolist(), '\t',
    #       'neg_loss: ', neg_loss.tolist(), '\t',
    #       'pos_acc: ', pos_acc.tolist(), '\t',
    #       'neg_acc ', neg_acc.tolist())

    print(
        'epoch: ', i, '  '
        'loss: ', loss.tolist(), '  ',
        'auprc: ', auprc, '  ',
        'auroc: ', auroc, '  ',
        'ap: ', ap, '  ',
        'p_acc ', pos_acc, '  ',
        'n_acc ', neg_acc
    )