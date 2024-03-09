#!/usr/bin/env python
# coding: utf-8

from ot_cond.utils import set_seed, get_G, get_dist, unfreeze, freeze
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
import ot
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from ot_cond.utils import createLogHandler
from os import getpid
from argparse import ArgumentParser
import random


parser = ArgumentParser()
logger = createLogHandler("logs/barycenter_exp_tabak.csv", str(getpid()))

parser.add_argument('--lambda_reg_x','-rx',default = 1e3,type = float)
parser.add_argument('--lambda_reg_y','-ry',default = 1e3,type = float)
parser.add_argument('--device','-d',default=0,type = int)
parser.add_argument('--noise_dim','-nd',default=10,type = int)
parser.add_argument('--n_x_noise','-nx',default=500,type = int)
parser.add_argument('--n_y_noise','-ny',default=1,type = int)
parser.add_argument('--noise_var_x','-vx',default=1,type = float)
parser.add_argument('--noise_var_y','-vy',default=1,type = float)
parser.add_argument('--hidden_dim','-hd',default=32,type = int)
parser.add_argument('--khp_x','-kx',default=1,type = float)
parser.add_argument('--khp_y','-ky',default=1,type = float)
parser.add_argument('--batch_size','-bs',default=100,type = int)
parser.add_argument('--n_epochs','-ne',default=1200,type = int)
parser.add_argument('--n_Z','-nz',default=200,type = int)
parser.add_argument('--lr','-l',type=float,default=1e-3)
parser.add_argument('--tcost_type','-tt',type=str,default='model')
parser.add_argument('--seed','-s',type=int,default=0)

args = parser.parse_args()

dtype = torch.float
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)



def m(Z):
    return 2*(Z-0.5)

def m_dash(Z):
    return -4*(Z-0.5)

def sig(Z):
    return 1

def sig_dash(Z):
    return 4

def best_map(x,z):
    return x+m_dash(z)-m(z)

d = 1
n_X = args.n_Z
n_Y = 1  # given X
n_noise = args.n_x_noise
n_ynoise = args.n_y_noise
NOISE_DIMENSION = args.noise_dim
GENERATOR_OUTPUT_IMAGE_SHAPE = d
n_epochs = args.n_epochs
lr = args.lr
batch_size = args.n_Z
lambda_reg_x = args.lambda_reg_x
lambda_reg_y = args.lambda_reg_y
khp = args.khp_x
P = 2
ktype = "rbf"
tcost_type = args.tcost_type

std_noise_x = np.sqrt(args.noise_var_x)
std_noise_y = np.sqrt(args.noise_var_y)


# In[3]:

def get_Xs(n_X):
    return (np.random.beta(2,4,size=(n_X, d)),np.random.beta(4,2,size=(n_X, d)))

def get_data(X,X_dash,n_Y=n_Y):
    
    if(type(X) == torch.Tensor):
        X = X.detach().cpu().numpy()
    if(type(X_dash) == torch.Tensor):
        X_dash = X_dash.detach().cpu().numpy()
    
    Y = []
    Y_dash = []

    for X_i,X_i_dash in zip(X,X_dash):
        Y_dash.append(np.random.multivariate_normal(m_dash(X_i_dash), np.eye(d)*sig_dash(X_i_dash), size=(n_Y,)))
        Y.append(np.random.multivariate_normal(m(X_i), np.eye(d)*sig(X_i), size=(n_Y,)))

    Y = np.array(Y)
    Y_dash = np.array(Y_dash)

    tensorX = torch.from_numpy(X).to(dtype).to(device)
    tensorX_dash = torch.from_numpy(X_dash).to(dtype).to(device)
    tensorY = torch.from_numpy(Y).to(dtype).to(device)
    tensorY_dash = torch.from_numpy(Y_dash).to(dtype).to(device)

    return tensorX, tensorX_dash, tensorY, tensorY_dash

##########################################################################################################
##########################################################################################################
########################################### MODELS #######################################################
##########################################################################################################
##########################################################################################################


"""
nns
""";
hidden_dim = args.hidden_dim

class GenY_X(nn.Module):
    def __init__(self, input_dim = d, out_dim = d):
        super(GenY_X, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(NOISE_DIMENSION+input_dim, hidden_dim, bias=True),
        # nn.BatchNorm1d(hidden_dim, 0.8),
        nn.LeakyReLU(0.25),
        # nn.SiLU(),
        nn.Linear(hidden_dim,hidden_dim, bias=True),
        nn.LeakyReLU(0.25),  
        # nn.SiLU(),
        # nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

class GenY_dash_XY(nn.Module):
    def __init__(self, input_dim = 2*d, out_dim = d):
        super(GenY_dash_XY, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(NOISE_DIMENSION+input_dim, hidden_dim, bias=True),
        # nn.BatchNorm1d(hidden_dim, 0.8),
        nn.LeakyReLU(0.25),
        # nn.SiLU(),
        nn.Linear(hidden_dim,hidden_dim, bias=True),
        # nn.SiLU(),
        nn.LeakyReLU(0.25),
        # nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


# In[5]:

# MODELS

input_dim = d
T = nn.Sequential(
        nn.Linear(2*input_dim, hidden_dim, bias=True),
        nn.LeakyReLU(0.1),
        nn.Linear(hidden_dim,GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
        # nn.BatchNorm1d(hidden_dim),
        # nn.LeakyReLU(0.1),
        # nn.Linear(hidden_dim, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
        ).to(device)
g = nn.Sequential(
        nn.Linear(2*input_dim, hidden_dim, bias=True),
        nn.LeakyReLU(0.1),
        nn.Linear(hidden_dim,GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
        # nn.BatchNorm1d(hidden_dim),
        # nn.LeakyReLU(0.1),
        # nn.Linear(hidden_dim, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
        ).to(device)
optimizer_T = Adam(T.parameters(), lr=lr)#,weight_decay=0.01)
optimizer_g = Adam(g.parameters(), lr=lr)#,weight_decay=0.01)


# ## Vectorized Version

# In[6]:


class BiDuplet(Dataset):
    def __init__(self,X,X_dash,Y,Y_dash):
        self.X = X
        self.X_dash = X_dash
        self.Y = Y
        self.Y_dash = Y_dash

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_dash[idx] ,self.Y[idx], self.Y_dash[idx]


# In[7]:


##########################################################################################################
##########################################################################################################
########################################## TRAINING ######################################################
##########################################################################################################
##########################################################################################################



def train_regr(dataloader, optimizer_T, optimizer_g, T, g, epochs, device = device, lambda_reg = lambda_reg_x, d=1, epochs_inner=5):
    ep_obj = []
    ws_obj = []
    # best_score = 1e10
    # best_model = None
    bs = 30
    test_z,test_z_ = get_Xs(bs)
    test_z = test_z*0 + 0.5
    test_z,test_z_,test_x,test_y = get_data(test_z,test_z)
    for t in tqdm(range(epochs)):
        tot_obj = 0.
        for b_idx, (z,z2,x,y) in enumerate(dataloader):
            m1 = x.shape[0]
            m2 = y.shape[0]
            x = x.view(m1,-1)
            y = y.view(m2,-1)
            unfreeze(g)
            for _ in range(epochs_inner):
                
                y_hat = T(torch.cat([x, z], dim=1))
                # print(y_hat.shape,x.shape)

                tcost = torch.trace(get_dist(y_hat, x))/m1

                g_score = g(torch.cat([y, z2], dim=1)).sum()

                log_gy_score = torch.log(torch.sum(torch.exp(g(torch.cat([y, z2], dim=1))),dim=0) +\
                                         torch.sum(torch.exp(g(torch.cat([y_hat, z], dim=1))),dim=0))
                
                obj = -((tcost + lambda_reg*g_score)/m1 - lambda_reg*log_gy_score + lambda_reg*np.log(2*m2))
                optimizer_g.zero_grad()
                obj.backward()
                optimizer_g.step()
            
            freeze(g)
            unfreeze(T)
            for _ in range(epochs_inner):
                y_hat = T(torch.cat([x, z], dim=1))

                tcost = torch.trace(get_dist(y_hat, x))/m1

                g_score = g(torch.cat([y, z2], dim=1)).sum()

                log_gy_score = torch.log(torch.sum(torch.exp(g(torch.cat([y, z2], dim=1))),dim=0) +\
                                         torch.sum(torch.exp(g(torch.cat([y_hat, z], dim=1))),dim=0))

                obj = (tcost + lambda_reg*g_score)/m1 - lambda_reg*log_gy_score + lambda_reg*np.log(2*m2)

                optimizer_T.zero_grad()
                obj.backward()
                optimizer_T.step()
                
            freeze(T)
            tot_obj = tot_obj + obj.item()
            # if t%1000 == 0:
            #     torch.save(T.state_dict(), f'tabak_model/decay/T-{lambda_reg}-{args.lr}-{args.i_epochs}-{t}.pt')
        
        test_y_gen = T(torch.cat([test_x.view(bs,-1), test_z], dim=1))
        
        tcost = torch.trace(get_dist(test_y_gen, test_y.view(bs,-1)))/bs
        
        ep_obj.append(tot_obj)
        ws_obj.append(tcost.item())
        # torch.save(T.state_dict(), f'tabak_model/decay/T-{lambda_reg}-{args.lr}-{args.i_epochs}-{epochs}.pt')
        # if(ws_dist<best_score):
        #     best_score = ws_dist
        #     best_model = deepcopy(T)
    return ep_obj,ws_obj , T, g


train_Z,train_Z_dash = get_Xs(n_X)
train_Z, train_Z_dash, train_X_Z, train_Y_Z = get_data(train_Z,train_Z_dash)
train_dataset = BiDuplet(train_Z, train_Z_dash, train_X_Z, train_Y_Z )
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)


obj,ws_obj, trained_T, trained_g = train_regr(train_dataloader,optimizer_T,optimizer_g,T,g,n_epochs,epochs_inner=40,lambda_reg=lambda_reg_x)

# print(f'cond no. y :{torch.linalg.cond(Gcat_y)}  cond no. x :{torch.linalg.cond(Gcat_x)}')

##########################################################################################################
##########################################################################################################
########################################## TESTING #######################################################
##########################################################################################################
##########################################################################################################


def average_Wasserstein_distance(XS,XT):
    total_distance = 0.
    assert len(XS) == len(XT)
    for i,(xs,xt) in enumerate(zip(XS,XT)):
        M = ot.dist(xs.view(-1,1),xt.view(-1,1))
        n_xs,n_xt = len(xs),len(xt)
        a, b = torch.ones((n_xs,)) / n_xs, torch.ones((n_xt,)) / n_xt
        total_distance += ot.emd2(a,b,M).item()
    return total_distance/len(XS)

def get_barycenter(xs,xt,c_type='random',bc_lambda=0.5):
    target_n_noise = xt.shape[-1]
    batch_size = xs.shape[0]
    if(c_type=='random'):
        bc_samples = bc_lambda*xs+(1-bc_lambda)*xt
    # bc_samples_sorted = bc_lambda*(torch.sort(x_Z_emperical)[0])+(1-bc_lambda)*torch.sort(y_samp_random)[0]
    elif(c_type=='repeat'):
        bc_samples = bc_lambda*xs.repeat_interleave(target_n_noise).view(batch_size,-1)+(1-bc_lambda)*xt
    elif(c_type=='mean'):
        bc_samples = bc_lambda*xs+(1-bc_lambda)*xt.mean(dim=2)
    return bc_samples
def get_barycenter_dual(s,t,s2t,t2s,bc_lambda=0.5):
    bs2t = bc_lambda*s+(1-bc_lambda)*s2t
    bt2s = (1-bc_lambda)*t+bc_lambda*t2s
    bs2t_split = int(bc_lambda*bs2t.shape[1])
    bt2s_split = int((1-bc_lambda)*bt2s.shape[1])
    return torch.cat([bs2t[:,:bs2t_split],bt2s[:,:bt2s_split]],dim=1)

def get_true_barycenter(Z,n_bc,t=0.5):
    if(type(Z)== torch.Tensor):
        Z = Z.detach().cpu().numpy()
    B = []
    for Z_i in Z:
        B.append(np.random.multivariate_normal((1-t)*m_dash(Z_i)+t*m(Z_i), np.eye(d)*(t*sig(Z_i)+(1-t)*sig_dash(Z_i)), size=(n_bc,)))
    B = np.array(B)
    return torch.from_numpy(B).to(dtype).to(device)


def barycenter_results(estimated_xt,Z):
    if(type== torch.Tensor):
        Z = Z.detach().cpu().numpy()
    estimated_mean = estimated_xt.mean(-1).view(-1).detach().cpu()
    true_mean = (m(Z)+m_dash(Z)).view(-1).detach().cpu()/2
    mean_abs_diff = torch.abs(estimated_mean-true_mean).mean().item()
    mean_diff = (estimated_mean-true_mean).mean().item()
    estimated_var = estimated_xt.var(-1).view(-1).detach().cpu().mean().item()
    return mean_abs_diff, mean_diff, estimated_var
    

n_testing_z = 30
testing_Z = torch.linspace(0,1,n_testing_z).view(-1,d)
testing_Z,testing_Z_,emperical_x,emperical_y = get_data(testing_Z,testing_Z,n_Y=n_noise)

testing_y = T(torch.cat([emperical_x.view(-1,d), testing_Z.repeat_interleave(n_noise).view(-1,d)], dim=1))
testing_y = testing_y.view(n_testing_z,-1)

bc_lambda = 0.5
emperical_centers = get_true_barycenter(testing_Z,n_noise,bc_lambda)
estimated_centers = bc_lambda*(emperical_x.view(n_testing_z,-1))+(1-bc_lambda)*testing_y.view(n_testing_z,-1)

wd_bc  = average_Wasserstein_distance(estimated_centers,emperical_centers)
wd_x = average_Wasserstein_distance(emperical_x,emperical_x)
wd_y = average_Wasserstein_distance(testing_y,emperical_y)
tcost = torch.diagonal(get_dist(emperical_x.view(1,-1,d),testing_y.view(1,-1,d)),offset=0,dim1=-1, dim2=-2).sum()/(n_noise*n_testing_z)

logger.info(f'{lambda_reg_x}, {lambda_reg_y}, {NOISE_DIMENSION}, {n_noise}, {n_ynoise}, {args.noise_var_x}, {args.noise_var_y}, {lr}, {hidden_dim},\
    {n_epochs}, {khp}, {khp}, {n_X}, {batch_size}, {SEED}, {tcost}, {obj[-1]}, {ws_obj[-1]}, {wd_x} , {wd_y}, {wd_bc}\
        ')
