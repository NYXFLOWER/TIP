from src.utils import *
from src.layers import *

# choose TIP model: 'cat' - TIP-cat
#					'add' - TIP-add
MOD = 'cat'
MAX_EPOCH = 100

# set training device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initial model
if MOD == 'cat':
	settings = Setting(sp_rate=0.9, lr=0.01, prot_drug_dim=16, n_embed=48, n_hid1=32, n_hid2=16, num_base=32)
	model = TIP(settings, device)
else:
	settings = Setting(sp_rate=0.9, lr=0.01, prot_drug_dim=64, n_embed=64, n_hid1=32, n_hid2=16, num_base=32)
	model = TIP(settings, device, mod='add')

# initial optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)

# train TIP model
for e in range(MAX_EPOCH):
	model.train()
	optimizer.zero_grad()
	loss = model()
	print(loss.item())
	loss.backward()
	optimizer.step()

# evaluate on test set
model.test()

# save trained model
torch.save(model, f'saved_model/tip-{model.mod}-example.pt')