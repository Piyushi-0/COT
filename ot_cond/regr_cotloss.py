"""
Author: Piyushi
"""
import torch
import numpy as np
from ot_cond.utils import get_dist
from ot_cond.utils import get_G_v3 as get_G, freeze, unfreeze
import itertools


def train(dataloader, f, pi_net, opt, n_noise, n_ynoise, batch_size, NOISE_DIMENSION, device, khp, lambda_reg, y_d, n_Y=1, ktype="rbf"):
    v2 = torch.cat([(1/(n_noise*n_ynoise))*torch.ones([n_noise*n_ynoise]), -(1/n_Y)*torch.ones([n_Y])]).to(device)
    
    f.train()
    pi_net.train()
    
    for _, (x, y) in enumerate(dataloader):
        n_X = x.shape[0]
        
        n_s = torch.randn(n_noise*batch_size, NOISE_DIMENSION).to(device)  # noise for f
        nx = torch.cat([n_s, x.repeat_interleave(n_noise, dim=0)], dim=1)  # noise.x concat for f
        gen_y = f(nx)  # geny
        
        xy = torch.cat([x.repeat_interleave(n_noise, dim=0), gen_y], dim=1)  # x.geny concat for pi_net
        noise_mat = torch.randn(n_ynoise*xy.shape[0], NOISE_DIMENSION).to(device)   # noise for pi_net
        final_y = pi_net(torch.cat([xy.repeat_interleave(n_ynoise, 0), noise_mat], dim=1))
        
        # transport cost
        tcost = torch.diagonal(get_dist(gen_y.repeat_interleave(n_ynoise).view(batch_size, -1, y_d)
                                      , final_y.view(batch_size,-1, y_d)), offset=0, dim1=-1, dim2=-2).sum()/(n_X*n_noise*n_ynoise)
        
        # mmd reg
        cat_y = torch.cat([final_y.view(batch_size, -1, y_d), y.view(batch_size, -1, y_d)], dim=1)
        Gcat_y = get_G(khp=khp, x=cat_y, y=cat_y, ktype=ktype)
        reg = torch.sum(v2@Gcat_y@v2)/n_X
        
        obj = tcost + lambda_reg*reg
        opt.zero_grad()
        obj.backward()
        opt.step()
        
    return obj, f, pi_net

def train_2side(dataloader, f, pi_net, opt, n_noise, n_ynoise, batch_size, NOISE_DIMENSION, device, khp1, khp2, lambda_reg, y_d, n_Y=1, ktype="rbf"):
    v1 = torch.cat([(1/n_noise)*torch.ones([n_noise]), -(1/n_Y)*torch.ones([n_Y])]).to(device)
    v2 = torch.cat([(1/(n_noise*n_ynoise))*torch.ones([n_noise*n_ynoise]), -(1/n_Y)*torch.ones([n_Y])]).to(device)
    
    f.train()
    pi_net.train()
    
    for _, (x, y, y_dash) in enumerate(dataloader):
        n_X = x.shape[0]
        
        n_s = torch.randn(n_noise*batch_size, NOISE_DIMENSION).to(device)  # noise for f
        nx = torch.cat([n_s, x.repeat_interleave(n_noise, dim=0)], dim=1)  # noise.x concat for f
        gen_y = f(nx)  # geny
        
        xy = torch.cat([x.repeat_interleave(n_noise, dim=0), gen_y], dim=1)  # x.geny concat for pi_net
        noise_mat = torch.randn(n_ynoise*xy.shape[0], NOISE_DIMENSION).to(device)   # noise for pi_net
        final_y = pi_net(torch.cat([xy.repeat_interleave(n_ynoise, 0), noise_mat], dim=1))
        
        # transport cost
        tcost = torch.diagonal(get_dist(gen_y.repeat_interleave(n_ynoise).view(batch_size, -1, y_d)
                                        , final_y.view(batch_size,-1, y_d)), offset=0, dim1=-1, dim2=-2).sum()/(n_X*n_noise*n_ynoise)
        
        # mmd reg
        cat_y = torch.cat([final_y.view(batch_size, -1, y_d), y.view(batch_size, -1, y_d)], dim=1)
        Gcat_y = get_G(khp=khp1, x=cat_y, y=cat_y, ktype=ktype)
        reg = torch.sum(v2@Gcat_y@v2)/n_X
        
        cat_x = torch.vstack([gen_y, y_dash])
        Gcat_x = get_G(khp=khp2, x=cat_x, y=cat_x, ktype=ktype)
        reg = reg + torch.sum(v1@Gcat_x@v1)/n_X
        
        obj = tcost + lambda_reg*reg
        opt.zero_grad()
        obj.backward()
        opt.step()
        
    return obj, f, pi_net

def train2_2side(dataloader1, dataloader2, f, pi_net, opt, n_noise, n_ynoise, batch_size, NOISE_DIMENSION, device, khp1, khp2, lambda_reg, y_d, n_Y=1, ktype="rbf"):
    v2 = torch.cat([(1/(n_noise*n_ynoise))*torch.ones([n_noise*n_ynoise]), -(1/n_Y)*torch.ones([n_Y])]).to(device)
    v1 = v2.clone()
    
    f.train()
    pi_net.train()
    
    dataloader2 = itertools.cycle(dataloader2)
    
    for _, (x, y) in enumerate(dataloader1):
        n_X = x.shape[0]

        n_s = torch.randn(n_noise*batch_size, NOISE_DIMENSION).to(device)  # noise for f
        nx = torch.cat([n_s, x.repeat_interleave(n_noise, dim=0)], dim=1)  # noise.x concat for f
        gen_y = f(nx)  # geny
        
        xy = torch.cat([x.repeat_interleave(n_noise, dim=0), gen_y], dim=1)  # x.geny concat for pi_net
        noise_mat = torch.randn(n_ynoise*xy.shape[0], NOISE_DIMENSION).to(device)   # noise for pi_net
        #import pdb; pdb.set_trace()
        final_y = pi_net(torch.cat([xy.repeat_interleave(n_ynoise, 0), noise_mat], dim=1))

        # transport cost
        tcost = torch.diagonal(get_dist(gen_y.repeat_interleave(n_ynoise).view(batch_size, -1, y_d)
                                      , final_y.view(batch_size, -1, y_d)), offset=0, dim1=-1, dim2=-2).sum()/(n_X*n_noise*n_ynoise)
        
        # mmd reg
        cat_y = torch.cat([final_y.view(batch_size, -1, y_d), y.view(batch_size, -1, y_d)], dim=1)
        Gcat_y = get_G(khp=khp1, x=cat_y, y=cat_y, ktype=ktype)
        reg = torch.sum(v2@Gcat_y@v2)/n_X
        
        ###---------------------------------------------------
        x_dash, y_dash = next(dataloader2)
        x_dash = x_dash.to(device)
        y_dash = y_dash.to(device)
        n_X = x_dash.shape[0]
        
        n_s2 = torch.randn(n_noise*batch_size, NOISE_DIMENSION).to(device)  # noise for f
        nx2 = torch.cat([n_s2, x_dash.repeat_interleave(n_noise, dim=0)], dim=1)  # noise.x concat for f
        gen_y2 = f(nx2)  # geny
        
        xy2 = torch.cat([x_dash.repeat_interleave(n_noise, dim=0), gen_y2], dim=1)  # x.geny concat for pi_net
        noise_mat = torch.randn(n_ynoise*xy2.shape[0], NOISE_DIMENSION).to(device)   # noise for pi_net
        final_y2 = pi_net(torch.cat([xy2.repeat_interleave(n_ynoise, 0), noise_mat], dim=1))
        
        # transport cost
        tcost = tcost + torch.diagonal(get_dist(gen_y2.repeat_interleave(n_ynoise).view(batch_size, -1, y_d)
                                    , final_y2.view(batch_size, -1, y_d)),offset=0,dim1=-1, dim2=-2).sum()/(n_X*n_noise*n_ynoise)
        
        cat_x = torch.cat([final_y2.view(batch_size, -1, y_d), y_dash.view(batch_size, -1, y_d)], dim=1)
        Gcat_x = get_G(khp=khp2, x=cat_x, y=cat_x, ktype=ktype)
        reg = reg + torch.sum(v1@Gcat_x@v1)/n_X
        
        obj = tcost + lambda_reg*reg
        opt.zero_grad()
        obj.backward()
        opt.step()
        
    return obj, f, pi_net

def tabak(dataloader1, dataloader2, optimizer_T, optimizer_g, T, g, epochs, device, lambda_reg, epochs_inner=100):
    def get_obj(x, z, y, z2, m1):
        y_hat = T(torch.cat([x, z], dim=1))
        tcost = torch.trace(get_dist(y_hat, x))/m1
        g_score = g(torch.cat([y, z2], dim=1)).sum()
        log_gy_score = torch.log(torch.sum(torch.exp(g(torch.cat([y, z2], dim=1)))) +\
                                    torch.sum(torch.exp(g(torch.cat([y_hat, z], dim=1)))))
        # NOTE: g's output will be of (num_samples, num_dim)
        obj = (tcost + lambda_reg*g_score)/m1 - lambda_reg*log_gy_score + lambda_reg*np.log(2*m2)
        return obj

    obj_g = []
    obj_T = []
    for i in range(epochs):
        tot_obj = 0.
        for b_i, (x, z) in enumerate(dataloader1):
            m1 = x.shape[0]
            x = x.to(device)
            z = z.to(device)
            
            y, z2 = next(dataloader2)
            m2 = y.shape[0]
            y = y.to(device)
            z2 = z2.to(device)
            
            # NOTE: this assumes m1=m2

            freeze(T)
            T.eval()
            
            unfreeze(g)
            g.train()
            
            for j in range(epochs_inner):
                obj = -get_obj(x, z, y, z2, m1)
                optimizer_g.zero_grad()
                obj.backward()
                optimizer_g.step()
                if i == epochs-1:
                    obj_g.append(-obj.item())
            ###
            
            freeze(g)
            g.eval()
            
            unfreeze(T)
            T.train()
            
            for j in range(epochs_inner):
                obj_t = get_obj(x, z, y, z2, m1)
                
                optimizer_T.zero_grad()
                obj_t.backward()
                optimizer_T.step()
                if i == epochs-1:
                    obj_T.append(obj_t.item())
                
            tot_obj = tot_obj + obj_t.item()
            
    return tot_obj, obj_g, obj_T, T, g
