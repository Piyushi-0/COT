from ot_cond.utils import set_seed, get_G, get_dist
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib
from torch.optim import Adam, SGD
import ot
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from matplotlib import gridspec
from tqdm import tqdm
import matplotlib.pylab as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from ot_cond.utils import createLogHandler
from argparse import ArgumentParser
from os import getpid
import random
import copy

parser = ArgumentParser()
parser.add_argument('--position','-pos',default=0.2,type = float)
args = parser.parse_args()



dtype = torch.float
device = torch.device('cuda:0')
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

font = {'weight' : 'bold' ,
        'size'   : 10}
matplotlib.rc('font', **font)
image_format = 'svg'



def m(Z):
    return 4*(Z-0.5)

def m_dash(Z):
    return -2*(Z-0.5)

def sig(Z):
    return 1+0*Z

def sig_dash(Z):
    return 8*Z+1

def optimal_map(X,Z):
    return m_dash(Z) + torch.sqrt(sig_dash(Z))*(X-m(Z))

def optimal_map_r(Y,Z):
    return m(Z) + torch.sqrt(sig(Z))*(Y-m_dash(Z))

d = 1
n_X = 600
n_Y = 1  # given X
n_noise = 500
n_ynoise = 1
NOISE_DIMENSION = 10
GENERATOR_OUTPUT_IMAGE_SHAPE = 1
n_epochs = 1500
lr = 0.01
batch_size = 50
# lambda_decay = 0.998
# lambda_reg = 10
# khp_y = 1e1
# khp_x = 1e1

lambda_reg_x = 500
lambda_reg_y = 500
lambda_decay = 1
khp = 1

"""
nns
""";
hidden_dim = 16
set_seed(SEED)

class GenY_X(nn.Module):
    def __init__(self, input_dim = d, out_dim = d):
        super(GenY_X, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(NOISE_DIMENSION+input_dim, hidden_dim, bias=True),
        # nn.BatchNorm1d(hidden_dim, 0.8),
        nn.LeakyReLU(0.1),
        # nn.SiLU(),
        nn.Linear(hidden_dim,GENERATOR_OUTPUT_IMAGE_SHAPE, bias=True),
        # nn.LeakyReLU(0.25),  
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
        # nn.LeakyReLU(0.25),
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

f_r = GenY_X()
f_r.apply(initialize_weights)
f_r = f_r.to(device)
f_r.train()

pi_net_r = GenY_dash_XY()
pi_net_r.apply(initialize_weights)
pi_net_r = pi_net_r.to(device)
pi_net_r.train()


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

            Gcat_y = get_G(khp=khp, x=cat_y, y=cat_y)
            reg2 = reg2 + torch.sum(v1@Gcat_y@v1)/n_X
            
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
            Gcat_y_ = get_G(khp=khp, x=cat_y_, y=cat_y_)
            reg1 = reg1 + torch.sum(v2@Gcat_y_@v2)/n_X
            # reg1 = torch.sum(v2@Gcat_y@v2)
            
            all_source_y = torch.cat([source_y.repeat_interleave(n_ynoise).view(batch_size,-1,d),
                               source_y_.repeat_interleave(n_ynoise).view(batch_size,-1,d)],dim=0)
            
            all_target_y = torch.cat([target_y.view(batch_size,-1,d),target_y_.view(batch_size,-1,d)],dim=0)
            
            # print('1st',source_y.shape,target_y.shape)
            # print('2nd',source_y_.shape,target_y_.shape)
            
            # print('all',all_source_y.shape,all_target_y.shape)
            
            tcost = tcost + torch.diagonal(get_dist(all_source_y,all_target_y),offset=0,dim1=-1, dim2=-2).sum()/(2*n_X*n_noise*n_ynoise)
            
            # tcost = tcost + torch.diagonal(get_dist(source_y_.view(batch_size,-1,d),target_y_.view(batch_size,-1,d)),offset=0,dim1=-1, dim2=-2).sum()/(n_X*n_noise*n_ynoise)
            # tcost = tcost*0
        #     print(tcost)
        # tcost = torch.diagonal(get_dist(x_samp.repeat_interleave(n_ynoise).view(batch_size,-1,d),
        #                        all_ysamp.view(batch_size, -1, d)),offset=0,dim1=-1, dim2=-2).sum()/(batch_size*n_noise*n_ynoise)

        #-----------------------------------------------------------------------------------------
        # print(reg1, reg2)
        # tcost, reg1 = tcost*0, reg1*0
        obj = tcost + lambda_reg_x*reg2 + lambda_reg_y*reg1
        # print(obj)

        opt.zero_grad()
        obj.backward()
        opt.step()

        tcosts.append(tcost.item())
        reg1s.append(reg1.item())
        reg2s.append(reg2.item())
        objs.append(obj.item())
        
    return tcosts, reg1s, reg2s, objs , Gcat_y, Gcat_y_




try:
    checkpoint = torch.load('checkpoints/checkpoint_dual_visualisation_.pth',map_location=device)

except:
    train_Z,train_Z_dash = get_Xs(n_X)
    train_Z, train_Z_dash, train_X_Z, train_Y_Z = get_data(train_Z,train_Z_dash)
    train_dataset = BiDuplet(train_Z, train_Z_dash, train_X_Z, train_Y_Z )
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
    train_dataset_r = BiDuplet(train_Z_dash, train_Z, train_Y_Z, train_X_Z )
    trian_dataloader_r = DataLoader(train_dataset_r,batch_size=batch_size,shuffle=False)
    
    optimizer = Adam(chain(f.parameters(), pi_net.parameters()), lr=lr)
    optimizer_r = Adam(chain(f_r.parameters(), pi_net_r.parameters()), lr=lr)
    
    tcosts, reg1s, reg2s, objs, Gcat_y, Gcat_x = get_obj_vectorized(train_dataloader,f,pi_net,optimizer,n_epochs=n_epochs)
    tcosts_r, reg1s_r, reg2s_r, objs_r, Gcat_y_r, Gcat_x_r = get_obj_vectorized(trian_dataloader_r,f_r,pi_net_r,optimizer_r,n_epochs=n_epochs)
    
    checkpoint = { 
    'epoch': n_epochs,
    'pi_net': pi_net.state_dict(),
    'f': f.state_dict(),
    'pi_net_r': pi_net_r.state_dict(),
    'f_r': f_r.state_dict(),
    'optimizer': optimizer.state_dict(),
    'optimizer_r': optimizer_r.state_dict(),
    }
    torch.save(checkpoint, 'checkpoints/checkpoint_dual_visualisation_.pth')
    
    checkpoint = torch.load('checkpoints/checkpoint_dual_visualisation_.pth',map_location=device)


pi_net.load_state_dict(checkpoint['pi_net'])
pi_net_r.load_state_dict(checkpoint['pi_net_r'])
f.load_state_dict(checkpoint['f'])
f_r.load_state_dict(checkpoint['f_r'])

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




##############################################################################################################
# BARYCENTER VISUALIZATIONS
##############################################################################################################


test_z,test_z_dash = get_Xs(500)
n_test_z = 10
# change pos below to different values between 0 and 1 to get different mappings of barycenter

test_z = test_z[:n_test_z]*0 + args.position
test_z,test_z_dash, tensorX_Z_test,tensorY_Z_test = get_data(test_z,test_z,n_Y = 1000)

tensorX_Z_test = tensorX_Z_test.view(n_test_z,-1,d)
tensorY_Z_test = tensorY_Z_test.view(n_test_z,-1,d)


batch_size= 10
n_s = torch.randn(n_noise*batch_size, NOISE_DIMENSION).to(device)
nzq = torch.cat([n_s, test_z.repeat_interleave(n_noise).view(-1,1)], dim=1)
x_samp = f(nzq)


#-----------------------------------------------------------------------------------------


xz_q = torch.cat([test_z.repeat_interleave(1000).view(-1,1), tensorX_Z_test.view(-1,1)], dim=1)
yz_q = torch.cat([test_z.repeat_interleave(1000).view(-1,1), tensorY_Z_test.view(-1,1)], dim=1)
noise_mat = torch.randn(n_ynoise*xz_q.shape[0], NOISE_DIMENSION).to(device)
# print(x_samp.shape)

all_ysamp = pi_net(torch.cat([xz_q.repeat_interleave(n_ynoise, 0), noise_mat], dim=1)).view(n_test_z,-1)
all_xsamp = pi_net_r(torch.cat([yz_q.repeat_interleave(n_ynoise, 0), noise_mat], dim=1)).view(n_test_z,-1)

all_ysamp_op = optimal_map(tensorX_Z_test,test_z.view(-1,1,d)).view(n_test_z,-1)
all_xsamp_op = optimal_map_r(tensorY_Z_test,test_z_dash.view(-1,1,d)).view(n_test_z,-1)


tensorX_Z_test = tensorX_Z_test.view(n_test_z,-1)
tensorY_Z_test = tensorY_Z_test.view(n_test_z,-1)

blue = np.array([0,0,1,1])
red = np.array([1,0,0,1])
interpolates = np.linspace(0,1,10)
mix_colors = blue*interpolates.reshape(10,1) + (1-interpolates).reshape(10,1)*red

fig, axs = plt.subplots(1,2,figsize=(10, 5))


for i, bc_lambda in enumerate(interpolates):
    bc_samples = get_barycenter_dual(tensorX_Z_test,tensorY_Z_test,all_ysamp,all_xsamp,bc_lambda=bc_lambda)
    bc_samples_op = get_barycenter_dual(tensorX_Z_test,tensorY_Z_test,all_ysamp_op,all_xsamp_op,bc_lambda=bc_lambda)
    
    sns.kdeplot(bc_samples[0].detach().cpu(),ax=axs[0],color = mix_colors[i])
    sns.kdeplot(bc_samples_op[0].detach().cpu(),ax=axs[1],color = mix_colors[i])

for i in range(2):
    sns.kdeplot(tensorY_Z_test[0].view(-1).cpu(),ax=axs[i],label=r'$t(\`y|x)$',color = red)
    sns.kdeplot(tensorX_Z_test[0].view(-1).cpu(),ax=axs[i],label=r'$s(y|x)$',color=blue)

    axs[i].set_xlim(-6,6)

    axs[i].legend()

axs[0].set_title('Proposed')
axs[1].set_title('Analytical')


plt.savefig(f'plots/barycenter_visualization.{image_format}', bbox_inches = 'tight',format = image_format ,pad_inches = 0.25,dpi=1200)

##############################################################################################################
# PLAN VISUALIZATIONS
##############################################################################################################


test_z,test_z_dash = get_Xs(500)
n_test_z = 10

all_x, all_y, all_z = [],[],[]
n_noise = 50
for z_pos in np.linspace(0.1,0.9,4):
    z_pos = round(z_pos,2)
    test_z = test_z[:n_test_z]*0 + z_pos
    test_z_b,test_z_dash_b, tensorX_Z_test_b, tensorY_Z_test_b = get_data(test_z,test_z,n_Y = n_noise)

    test_z, test_z_dash = test_z_b.clone(),test_z_dash_b.clone()
    tensorX_Z_test, tensorY_Z_test = tensorX_Z_test_b.clone() , tensorY_Z_test_b.clone()

    batch_size= 10
    n_s = torch.randn(n_noise*batch_size, NOISE_DIMENSION).to(device)
    nzq = torch.cat([n_s, test_z.repeat_interleave(n_noise).view(-1,1)], dim=1)
    x_samp = f(nzq).view(n_test_z,-1)
    y_samp = f(nzq).view(n_test_z,-1)

    #-----------------------------------------------------------------------------------------
    xz_q = torch.cat([test_z.repeat_interleave(n_noise).view(-1,1), tensorX_Z_test.view(-1,1)], dim=1)
    yz_q = torch.cat([test_z.repeat_interleave(n_noise).view(-1,1), tensorY_Z_test.view(-1,1)], dim=1)

    xz_q_gen = torch.cat([test_z.repeat_interleave(n_noise).view(-1,1), x_samp.view(-1,1)], dim=1)
    yz_q_gen = torch.cat([test_z.repeat_interleave(n_noise).view(-1,1), y_samp.view(-1,1)], dim=1)

    noise_mat = torch.randn(n_ynoise*xz_q.shape[0], NOISE_DIMENSION).to(device)

    all_ysamp = pi_net(torch.cat([xz_q.repeat_interleave(n_ynoise, 0), noise_mat], dim=1)).view(n_test_z,-1)
    all_xsamp = pi_net_r(torch.cat([yz_q.repeat_interleave(n_ynoise, 0), noise_mat], dim=1)).view(n_test_z,-1)

    gen_ysamp =  pi_net(torch.cat([xz_q_gen.repeat_interleave(n_ynoise, 0), noise_mat], dim=1)).view(n_test_z,-1)
    gen_xsamp =  pi_net(torch.cat([yz_q_gen.repeat_interleave(n_ynoise, 0), noise_mat], dim=1)).view(n_test_z,-1)

    tensorX_Z_test = tensorX_Z_test.view(n_test_z,-1)
    tensorY_Z_test = tensorY_Z_test.view(n_test_z,-1)


    blue = np.array([0,0,1,1])
    red = np.array([1,1,0,1])

    interpolates = np.linspace(0,1,10)
    mix_colors = blue*interpolates.reshape(10,1) + (1-interpolates).reshape(10,1)*red

    xx,yy = x_samp.view(-1).detach().cpu().numpy(),gen_ysamp.view(-1).detach().cpu().numpy()
    
    all_x += list(xx)
    all_y += list(yy)
    all_z += [z_pos]*len(xx)

df = pd.DataFrame({'x':all_x,'y':all_y,'z':all_z});
df.rename(columns = {'z':''}, inplace = True)


plt.figure(figsize=(10,8))

gs = gridspec.GridSpec(3, 3)
ax1 = pl.subplot(gs[0, 1:])
source_plot = sns.kdeplot(x=df.x,hue=df[''],levels=3,legend=False,palette=sns.dark_palette("#00f", as_cmap=True))
ax1.set_xlabel('')
ax1.set_yticks([])
ax1.set_xticks([])
ax1.set_ylabel('')
ax1.set_title('Source Distributions')


ax2 = pl.subplot(gs[1:, 0])
target_plot = sns.kdeplot(y=df.y,hue=df[''],ax=ax2,levels=7,legend=False,palette=sns.dark_palette("#00f", as_cmap=True))

ax2.invert_yaxis()
ax2.invert_xaxis()
                          
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('')
ax2.set_ylabel('Target Distributions')       

                          
ax = pl.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)

plan_plot = sns.kdeplot(x=df.x,y=df.y,hue=df[''],levels=4,legend=True,ax=ax,palette=sns.dark_palette("#00f", as_cmap=True))

plt.savefig(f'plots/plan_plot__.{image_format}', bbox_inches = 'tight',format = image_format ,pad_inches = 0.25,dpi=1200)

