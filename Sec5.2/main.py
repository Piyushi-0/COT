# Imports

import argparse
import os
from itertools import chain

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import torch
from ot_cond.utils import createLogHandler, eye_like, get_dist
from ot_cond.utils import get_G_v2 as get_G
from ot_cond.utils import initialize_weights, set_seed
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Lambda, ToTensor
from tqdm import tqdm

from bio_utils import (compute_drug_signature_differences, compute_scalar_mmd,
                       compute_wasserstein_loss)
from data import get_data
from model import Generator

### Device and set seed

dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

SEED = 0
set_seed(SEED)

def save_model(pi_net, optimizer, epoch):
    torch.save(pi_net.state_dict(),f'{save_as}/model_trained_{epoch}.pth')
    torch.save(optimizer.state_dict(),f'{save_as}/optimizer_trained_{epoch}.pth')
    return

### Hyperparams

parser = argparse.ArgumentParser(description = '_')
parser.add_argument('--lda', type=float, default=2000) 
parser.add_argument('--ktype', type=str, default="imq")
parser.add_argument('--khp_x', type=float, default=10)
parser.add_argument('--khp_y', type=float, default=10)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--noise_dim', type=int, default=10)
parser.add_argument('--n_noise', type=int, default=20)
parser.add_argument('--opti_type', type=str, default="adam")
parser.add_argument('--log_fname', type=str, default="logs_final_run")
parser.add_argument('--save_as', required=False, type=str, default="")
parser.add_argument('--holdout', required=False, type=int, default=0)

args = parser.parse_args()
print(args)

# Hyperparameters
lambda_reg = args.lda
ktype = args.ktype
khp_x = args.khp_x
khp_y = args.khp_y
num_epochs = args.n_epochs
lr = args.lr
opti_type = args.opti_type
save_as = f"{lambda_reg}_{ktype}_{khp_x}_{khp_y}_{args.n_noise}_{args.holdout}" + args.save_as
n_noise = args.n_noise
noise_dim = args.noise_dim
holdout = args.holdout

if holdout == 0:
    holdout = None

logger = createLogHandler(f'{args.log_fname}.csv')
logger.info('--new run--')
logger.info("SEED, save_as, lambda_reg, ktype, khp_x, khp_y, num_epochs, lr")
logger.info(f"{SEED}, {save_as}, {lambda_reg}, {ktype}, {khp_x}, {khp_y}, {num_epochs}, {lr}")

# init logging
os.makedirs(save_as, exist_ok=True)

def train(pi_net, optimizer, X_train_dataloader, Y_train_dataloaders, scheduler=None):
    objs = []
    tcosts = []
    reg_costs = []
    best_obj = None # NOTE: ADDED
    v_gen = torch.Tensor(n_noise*[1.0/n_noise]).to(dtype).to(device)
    
    # Z_qs = torch.Tensor([10,100,1000,10000]).to(dtype).to(device) 
    Z_qs = torch.Tensor(list(Y_train_dataloaders.keys())).to(dtype).to(device)
    print(Z_qs)

    # TODO: TEST THE ABOVE CHANGE
    
    # TODO: if out of sample, quit on the required context
    # Notel should not be required then

    for epoch in tqdm(range(num_epochs)):
        obj_epoch = 0.
        tcost_epoch = 0.
        reg_epoch = 0.
        pi_net.train()
        
        
        for idx, X_ in enumerate(X_train_dataloader):
            X = X_[0].to(device) # pytorch dataloader returns list, but we want only the first element
            tcost = 0.
            reg_y = 0.
            obj = 0.
            # now loop over all Z:
            n_x = len(X)
            for ctx, dl in Y_train_dataloaders.items():
                ctx = torch.Tensor([ctx]).to(device)
                Y_Z_ = next(iter(dl))
                Y_Z = Y_Z_[0].to(device)

                ZX = torch.cat([X, ctx.repeat(n_x,1)], dim=1)
                n_s = torch.randn(n_noise*n_x, noise_dim).to(device)
                ZXn = torch.cat([ZX.repeat_interleave(n_noise,0), n_s],dim=1)

                k = Y_Z.shape[0]
                v_gen = torch.Tensor(ZXn.shape[0]*[1.0/ZXn.shape[0]]).to(dtype).to(device)
                
                v_ygz = -1*torch.Tensor(k*[1.0/k]).to(dtype).to(device)
                v = torch.cat([v_gen, v_ygz], dim=0).to(dtype).to(device)
                
                Y_gen = pi_net(ZXn)
                
                C = get_dist(X, Y_gen)
                
                reqd_indices = torch.eye(C.shape[0]).to(device).repeat_interleave(n_noise,dim=1)
                C = C*reqd_indices / 100
                # C = F.normalize(C, p=float('inf'), dim=[0,1])
                
                cat_Y = torch.vstack([Y_gen, Y_Z])
                
                #v = (1/cat_Y.shape[0])*torch.ones(cat_Y.shape[0]).to(dtype).to(device)
                G_cat = get_G(khp=khp_y, x=cat_Y, y=cat_Y, ktype=ktype)
                
                #print(G_cat.shape, v.shape)
                reg_y = reg_y + torch.mv(G_cat, v).dot(v)
                tcost = tcost + torch.sum(C)/(len(Z_qs)*(ZXn.shape[0]))
            
            obj = (tcost + lambda_reg*reg_y)/4
            
            # optimizer updates after looping for all Z
            optimizer.zero_grad()
            obj.backward()
            optimizer.step()
            
            obj_epoch += obj
            tcost_epoch += tcost
            reg_epoch += reg_y
        #if epoch == 0:
        #    print(f'Cond No.{torch.linalg.cond(G_cat)}')
        #print(f'Tcost: {tcost_epoch}, reg_cost: {reg_epoch}, Objective: {obj_epoch}')
        
        objs.append(obj_epoch)
        tcosts.append(tcost_epoch)
        reg_costs.append(reg_epoch)
        print(obj_epoch, tcost_epoch, reg_epoch)
        if scheduler:
            scheduler.step()

        if epoch % 50 == 0 and (best_obj is None or best_obj > objs[-1].item()):
            #cond_no = torch.linalg.cond(G_cat)
            #print(cond_no)
            best_obj = objs[-1].item()
            save_model(pi_net, optimizer, epoch)
            logger.info(f'{obj_epoch}, {tcost_epoch}, {reg_epoch}, {epoch}')

    return pi_net, optimizer

### Instantiate

def main():
    print('Loading Data')
    print('Holdout Value:', holdout)
    X_train_dataloader, Y_train_dataloaders, X_test_dataloader, Y_test_dataloaders =  get_data(filename='./datasets/hvg.h5ad', holdout=holdout)
    pi_net = Generator(noise_dim=noise_dim)
    pi_net.apply(initialize_weights)
    pi_net = pi_net.to(device)
    pi_net.train()

    optimizer = Adam(pi_net.parameters(), lr=lr, betas=(0.9,0.99))
    # scheduler = StepLR(optimizer, step_size=200, gamma=0.8)
    scheduler = None

    train(pi_net, optimizer, X_train_dataloader,Y_train_dataloaders, scheduler)
    save_model(pi_net, optimizer, num_epochs)

main()
