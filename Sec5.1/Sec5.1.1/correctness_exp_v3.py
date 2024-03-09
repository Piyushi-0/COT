#!/usr/bin/env python
# coding: utf-8

from ot_cond.utils import set_seed, get_G, get_dist
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
logger = createLogHandler("logs/correctness_exp_full_v3.csv", str(getpid()))

parser.add_argument('--lambda_reg_x','-rx',default = 5000,type = float)
parser.add_argument('--lambda_reg_y','-ry',default = 5000,type = float)
parser.add_argument('--device','-d',default=0,type = int)
parser.add_argument('--noise_dim','-nd',default=10,type = int)
parser.add_argument('--n_x_noise','-nx',default=500,type = int)
parser.add_argument('--n_y_noise','-ny',default=1,type = int)
parser.add_argument('--noise_var_x','-vx',default=1,type = float)
parser.add_argument('--noise_var_y','-vy',default=1,type = float)
parser.add_argument('--hidden_dim','-hd',default=16,type = int)
parser.add_argument('--khp_x','-kx',default=3,type = float)
parser.add_argument('--khp_y','-ky',default=3,type = float)
parser.add_argument('--batch_size','-bs',default=50,type = int)
parser.add_argument('--n_epochs','-ne',default=2000,type = int)
parser.add_argument('--n_Z','-nz',default=200,type = int)
parser.add_argument('--lr','-l',type=float,default=0.008)
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
    return 4*(Z-0.5)

def m_dash(Z):
    return -2*(Z-0.5)

def sig(Z):
    return 1

def sig_dash(Z):
    return 8*Z+1

d = 1
n_X = args.n_Z
n_Y = 1  # given X
n_noise = args.n_x_noise
n_ynoise = args.n_y_noise
NOISE_DIMENSION = args.noise_dim
GENERATOR_OUTPUT_IMAGE_SHAPE = d
n_epochs = args.n_epochs
lr = args.lr
batch_size = args.batch_size
# lambda_decay = 0.998
# lambda_reg = 10
# khp_y = 1e1
# khp_x = 1e1

lambda_reg_x = args.lambda_reg_x
lambda_reg_y = args.lambda_reg_y
lambda_decay = 1
khp_x = args.khp_x
khp_y = args.khp_y


def get_Xs(n_X):
    # return (np.random.uniform(size=(n_X,d)),np.random.uniform(size=(n_X,d)))
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
        nn.LeakyReLU(0.1),
        # nn.SiLU(),
        nn.Linear(hidden_dim,GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
        # nn.LeakyReLU(0.1),  
        # nn.SiLU(),
        # nn.Dropout(p=0.2),
        # nn.Linear(hidden_dim, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
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
        nn.LeakyReLU(0.1),
        # nn.SiLU(),
        nn.Linear(hidden_dim,GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
        # nn.SiLU(),
        # nn.LeakyReLU(0.1),
        # nn.Dropout(p=0.2),
        # nn.Linear(hidden_dim, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
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

f = GenY_X()
f.apply(initialize_weights)
f = f.to(device)
f.train()

pi_net = GenY_dash_XY()
pi_net.apply(initialize_weights)
pi_net = pi_net.to(device)
pi_net.train()

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

def get_obj_vectorized(dataloader,f,pi_net,opt,batch_size=batch_size,n_epochs=n_epochs):
    
    v1 = torch.cat([(1/n_noise)*torch.ones([n_noise]), -(1/n_Y)*torch.ones([n_Y])]).to(device)
    
    v2 = torch.cat([(1/(n_noise*n_ynoise))*torch.ones([n_noise*n_ynoise]), -(1/n_Y)*torch.ones([n_Y])]).to(device)
    
    # best_model = None
    # best_score = 1e20
    reg1s = []
    reg2s = []
    objs = []
    tcosts = []
    f.train()
    pi_net.train()
    
    for e in tqdm(range(n_epochs)):
        reg2 = 0.
        reg1 = 0.
        tcost = 0.
        
        for q, (x,x_,Y,Y_) in enumerate(dataloader):

            n_s = torch.randn(n_noise*batch_size, NOISE_DIMENSION).to(device)
            nx = torch.cat([n_s, x.repeat_interleave(n_noise).view(-1,1)], dim=1)
            source_y = f(nx)
            
            cat_y = torch.cat([source_y.view(batch_size,-1,d),Y.view(batch_size,-1,d)],dim=1)
            # print(cat_y.shape)
            # return
            Gcat_y = get_G(khp=khp_y, x=cat_y, y=cat_y)
            reg2 = reg2 + torch.sum(v1@Gcat_y@v1)/n_X
            # reg2 = torch.sum(v1@Gcat_x@v1)
            
            xy = torch.cat([x.repeat_interleave(n_noise).view(-1,1), source_y], dim=1)
            noise_mat = torch.randn(n_ynoise*xy.shape[0], NOISE_DIMENSION).to(device)
            target_y = pi_net(torch.cat([xy.repeat_interleave(n_ynoise, 0), noise_mat], dim=1))


            #-----------------------------------------------------------------------------------------
            
            n_s = torch.randn(n_noise*batch_size, NOISE_DIMENSION).to(device)
            nx_ = torch.cat([n_s, x_.repeat_interleave(n_noise).view(-1,1)], dim=1)
            source_y_ = f(nx_)
            
            x_y_ = torch.cat([x_.repeat_interleave(n_noise).view(-1,1), source_y_], dim=1)
            noise_mat = torch.randn(n_ynoise*source_y_.shape[0], NOISE_DIMENSION).to(device)
            target_y_ = pi_net(torch.cat([x_y_.repeat_interleave(n_ynoise, 0), noise_mat], dim=1))
            
            cat_y_ = torch.cat([target_y_.view(batch_size,-1,d), Y_.view(batch_size,-1,d)],dim=1)
            Gcat_y_ = get_G(khp=khp_x, x=cat_y_, y=cat_y_)
            reg1 = reg1 + torch.sum(v2@Gcat_y_@v2)/n_X
            # reg1 = torch.sum(v2@Gcat_y@v2)
            
            all_source_y = torch.cat([source_y.repeat_interleave(n_ynoise).view(batch_size,-1,d),
                               source_y_.repeat_interleave(n_ynoise).view(batch_size,-1,d)],dim=0)
            
            all_target_y = torch.cat([target_y.view(batch_size,-1,d),target_y_.view(batch_size,-1,d)],dim=0)
            
            
            tcost = tcost + torch.diagonal(get_dist(all_source_y,all_target_y),offset=0,dim1=-1, dim2=-2).sum()/(2*n_X*n_noise*n_ynoise)
            
            # tcost = tcost + torch.diagonal(get_dist(source_y_.view(batch_size,-1,d),target_y_.view(batch_size,-1,d)),offset=0,dim1=-1, dim2=-2).sum()/(n_X*n_noise*n_ynoise)
            # tcost = tcost*0
        #     print(tcost)
        # tcost = torch.diagonal(get_dist(x_samp.repeat_interleave(n_ynoise).view(batch_size,-1,d),
        #                        all_ysamp.view(batch_size, -1, d)),offset=0,dim1=-1, dim2=-2).sum()/(batch_size*n_noise*n_ynoise)

        #-----------------------------------------------------------------------------------------

        # tcost, reg1 = tcost*0, reg1*0
        obj = tcost + lambda_reg_x*reg2 + lambda_reg_y*reg1

        opt.zero_grad()
        obj.backward()
        opt.step()

        tcosts.append(tcost.item())
        reg1s.append(reg1.item())
        reg2s.append(reg2.item())
        objs.append(obj.item())
        
        
        
    return tcosts, reg1s, reg2s, objs , Gcat_y, Gcat_y_
        
           
        

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

train_Z,train_Z_dash = get_Xs(n_X)
train_Z, train_Z_dash, train_X_Z, train_Y_Z = get_data(train_Z,train_Z_dash)
train_dataset = BiDuplet(train_Z, train_Z_dash, train_X_Z, train_Y_Z )
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

optimizer = Adam(chain(f.parameters(), pi_net.parameters()), lr=lr)

tcosts, reg1s, reg2s, objs, Gcat_y, Gcat_x = get_obj_vectorized(train_dataloader,f,pi_net,optimizer,n_epochs=n_epochs)

for testing_z in np.arange(0.1,1,0.1):
    
    testing_Z = torch.tensor(testing_z,dtype = dtype).view(-1,1).to(device)
    testing_Z,testing_Z_dash,emperical_x,emperical_y = get_data(testing_Z,testing_Z,n_Y=n_noise)
        
    for _ in range(20):
        
        n_s = torch.randn(n_noise, NOISE_DIMENSION).to(device)
        testing_zq = torch.cat([n_s, testing_Z.repeat_interleave(n_noise).view(-1,1)], dim=1)
        testing_x = f(testing_zq)
        testing_xz = torch.cat([testing_Z.repeat_interleave(n_noise).view(-1,1), testing_x], dim=1)
        testing_noise_mat = torch.randn(n_ynoise*testing_xz.shape[0], NOISE_DIMENSION).to(device)
        testing_y = pi_net(torch.cat([testing_xz.repeat_interleave(n_ynoise, 0), testing_noise_mat], dim=1))


        # tcost = torch.sum(get_dist(testing_x.view(n_noise, 1, d),
        #                            testing_y.view(n_noise, n_ynoise, d)))/(n_noise*n_ynoise)

        tcost = torch.diagonal(get_dist(testing_x.view(1,-1,d),testing_y.view(1,-1,d)),offset=0,dim1=-1, dim2=-2).sum()/(n_noise*n_ynoise)

        wd_x = average_Wasserstein_distance(testing_x.view(1,-1),emperical_x)
        wd_y = average_Wasserstein_distance(testing_y.view(1,-1),emperical_y)


        logger.info(f'{lambda_reg_x}, {lambda_reg_y}, {NOISE_DIMENSION}, {n_noise}; {n_ynoise}, {args.noise_var_x},{args.noise_var_y}, {lr}, {hidden_dim},\
            {n_epochs}, {khp_x}, {khp_y}, {n_X}, {batch_size}, {args.seed}, {testing_z}, {tcost}, {reg1s[-1]}, {reg2s[-1]}, {wd_x} , {wd_y} \
                ')
