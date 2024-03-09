# Imports
import anndata
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import torch
from ot_cond.utils import createLogHandler, get_dist
from ot_cond.utils import get_G_v2 as get_G
from ot_cond.utils import set_seed
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Lambda, ToTensor

# Lifted from CellOT
def transport_cot(pi_net, batch, dose, n_noise=3):
    n_x = len(batch)
    n_noise = n_noise
    noise_dim = 10
    ZX = torch.cat([batch, torch.Tensor([dose]).repeat(n_x,1)], dim=1).to(device)
    n_s = torch.randn(n_noise, noise_dim).to(device)
    ZXn = torch.cat([ZX.repeat_interleave(n_noise,0), n_s.repeat(n_x,1)],dim=1).to(device)
            
    with torch.no_grad():
        gen_batch = pi_net(ZXn).detach().cpu()

    return gen_batch


def compute_drug_signature_differences(control, treated, pushfwd):
    base = control.mean(0)

    true = treated.mean(0)
    pred = pushfwd.mean(0)

    diff = true - pred
    return diff

def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()
 
def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))

def compute_wasserstein_loss(x, y, epsilon=0.1):
    """Computes transport between x and y via Sinkhorn algorithm."""
    a = jnp.ones(len(x)) / len(x)
    b = jnp.ones(len(y)) / len(y)

    # compute cost
    geom_xy = pointcloud.PointCloud(x, y, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom_xy,a,b)
    # solve ot problem
    solver = sinkhorn.Sinkhorn()
    #out_xy = sinkhorn.Sinkhorn(geom_xy, a, b, max_iterations=100, min_iterations=10)
    out = solver(prob)
    # return regularized ot cost
    return out.reg_ot_cost
