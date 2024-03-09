import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from ot_cond.cotloss import train_pi_v2
from ot_cond.utils import (
    createLogHandler,
    get_auc_acc,
    get_dist,
    get_G_v2,
    initialize_weights,
    set_seed,
)
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import normalize
from torch.optim import SGD, Adagrad, Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import ClassifierMLP, NeuralMap


parser = argparse.ArgumentParser(description = "_")
parser.add_argument('--lda', type=float, default=10.0)
parser.add_argument('--ktype', type=str, default="imq")
parser.add_argument('--khp', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--wd', type=float, default=5e-7)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--save_as', default="test")

args = parser.parse_args()

lda = args.lda
ktype = args.ktype
khp = args.khp
lr = args.lr
wd = args.wd
n_epochs = args.n_epochs
save_as = args.save_as

opti_type = "adam"
to_plot = 0
del_prv = 0
log_fname = "logs"


n_piepochs = 200 // n_epochs
n_fepochs = 200 // n_epochs

SEED = 0
set_seed(SEED)

lambda_reg = lda

os.makedirs(save_as, exist_ok=True)
if del_prv:
    os.remove(f"{log_fname}.csv")

logger = createLogHandler(f"{log_fname}.csv", "1")
logger.info("---new run---")

logger.info("SEED, save_as, lambda_reg, ktype, khp, n_epochs, lr, opti_type, log_fname")
logger.info(f"{SEED}, {save_as}, {lambda_reg}, {ktype}, {khp}, {n_epochs}, {lr}, {opti_type}, {log_fname}")

device = torch.device("cuda")
dtype = torch.float32

data = loadmat("../../../data/awa_10_pc.mat")  # for training, sample testing

wembs = loadmat("../../../data/awa_wembs.mat")
wembs_s = torch.from_numpy(normalize(wembs['Xs'])).to(dtype).to(device)  # FIXED TO MATCH THEIRS
wembs_t = torch.from_numpy(normalize(wembs['Xt'])).to(dtype).to(device)  # FIXED TO MATCH THEIRS

X_train_orig = data['Xtr']
Y_train_orig = data['Ytr']

labels = np.where(Y_train_orig)[1] 


X_test = np.load('../../../../data/awaX_test.npy')
Y_test = np.load('../../../../data/awaY_test.npy')-1

d, n_cls = X_test.shape[1], 50

batch_size = 150  # FIXED TO MATCH THEIRS

def get_t(a):
    return torch.from_numpy(a).to(dtype).to(device)


#train_indices, val_indices =  torch.utils.data.random_split(range(len(X_train_orig)),[400,100] )
train_idx, val_idx = train_test_split(range(len(X_train_orig)),train_size=0.8,stratify=labels)
#train_idx = [index for index in train_indices ]
#val_idx = [index for index in val_indices ]

X_train, Y_train = X_train_orig[train_idx].copy(), Y_train_orig[train_idx].copy()
X_val, Y_val = X_train_orig[val_idx].copy(), Y_train_orig[val_idx].copy()

X_train, X_val, Y_train, Y_val = get_t(X_train), get_t(X_val), get_t(Y_train), get_t(Y_val)
print(X_train.shape)
val_dataset = TensorDataset(X_val, Y_val.to(torch.long))
val_dataloader = DataLoader(val_dataset, batch_size = len(X_val), shuffle = True)
    
train_dataset = TensorDataset(X_train, Y_train.to(torch.long))
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


test_dataset = TensorDataset(get_t(X_test), get_t(Y_test).to(torch.long))
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

###
### Cost and Gram Matrix
###

C = torch.cdist(wembs_s, wembs_t) ** 2  # FIXED TO MATCH THEIRS
if ktype == "lin":
    G = get_G_v2(ktype="lin", x=wembs_s, y=wembs_t)
elif ktype in ["rbf", "imq", "imq_v2"]:
    G = get_G_v2(dist=C, khp=khp, ktype=ktype)

"""
initialize networks & optimizer
"""
f = ClassifierMLP(d, n_cls)
f = f.to(device)
# f.fc1.weight.data = nn.Parameter(torch.ones_like(f.fc1.weight.data))  # TODO: can change
f.train()

pi_net = NeuralMap(input_dim=d + n_cls, out_dim=n_cls)
pi_net.apply(initialize_weights)
pi_net = pi_net.to(device)
pi_net.train()

scheduler = None
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs * len(train_dataloader))

"""
training & evaluation
"""
t_loss = []
r_loss = []

best_auc = None
best_acc = None

load_prv = 0

if load_prv:
    try:
        checkpoint = torch.load(f"{save_as}/best_val_{log_fname}.pt")

        best_auc = checkpoint["best_auc"]
        best_acc = checkpoint["best_acc"]

        f.load_state_dict(checkpoint["model_state_dict"])
        f.train()
    except:
        pass


optimizer_f = Adam(f.parameters(), lr=lr)
optimizer_pi = Adam(pi_net.parameters(), lr=lr, weight_decay=wd)

for t in range(n_epochs):
    print("Outer Epoch", t)
    for t_pi in tqdm(range(n_piepochs)):
        to_train = "pi"
        pi, f, tloss, rloss, optimizer_pi, scheduler = train_pi_v2(
            to_train,
            train_dataloader,
            optimizer_pi,
            None,
            f,
            pi_net,
            C,
            G,
            lambda_reg,
            bsize=batch_size,
            d=d,
            SEED=SEED,
            y_as_labels=0,
        )

    for t_f in range(n_fepochs):
        to_train = "f"
        pi_net, f, tloss, rloss, optimizer_f, scheduler = train_pi_v2(to_train,train_dataloader,optimizer_f,None,f,pi_net,C,G,lambda_reg,bsize=batch_size,d=d,SEED=SEED,y_as_labels=0)

    t_loss.append(tloss.item())
    r_loss.append(rloss.item())
    acc, auc = get_auc_acc(val_dataloader, f, device, dtype, y_as_labels=0)
    train_acc, train_auc = get_auc_acc(train_dataloader, f, device, dtype, y_as_labels=0)
    print("Training:", train_acc, train_auc)
    print("Val:", acc, auc)

    if best_acc is None or acc > best_acc:
        best_auc = auc
        best_acc = acc
        logger.info(f"{t}, {best_acc}, {best_auc}")
        torch.save(
            {
                "model_state_dict": f.state_dict(),
                "best_auc": best_auc,
                "best_acc": best_acc,
                "optimizer_f": optimizer_f.state_dict(),
                "optimizer_pi": optimizer_pi.state_dict()
            },
            f"{save_as}/best_val_{log_fname}.pt",
        )

print(f"Best_AUC={best_auc}_ACC={best_acc}")

t_loss = np.array(t_loss)
r_loss = np.array(r_loss)

plt.clf()
plt.plot(t_loss + r_loss)
plt.savefig(f"{save_as}/val_loss_{log_fname}.jpg")

plt.clf()
plt.plot(t_loss)
plt.savefig(f"{save_as}/val_tloss_{log_fname}.jpg")

plt.clf()
plt.plot(r_loss)
plt.savefig(f"{save_as}/val_rloss_{log_fname}.jpg")

checkpoint = torch.load(f"{save_as}/best_val_{log_fname}.pt")
f.load_state_dict(checkpoint["model_state_dict"])
f.eval()
acc, auc = get_auc_acc(train_dataloader, f, device, dtype, y_as_labels=0)
logger.info(f"Train_Acc_{acc}, Train_AUC_{auc}")
acc, auc = get_auc_acc(test_dataloader, f, device, dtype, y_as_labels=1)
logger.info(f"Test_Acc_{acc}, Test_AUC_{auc}")
logger.info('--End Run--')
