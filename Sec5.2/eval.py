## Imports
import argparse
import json
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
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Lambda, ToTensor
from tqdm import tqdm

from bio_utils import (compute_drug_signature_differences, compute_scalar_mmd,
                       compute_wasserstein_loss)
from data import get_data
from model import Generator

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# SEED = 0
# set_seed(SEED)


parser = argparse.ArgumentParser(description="_")
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--noise_dim", type=int, default=5)
parser.add_argument("--log_fname", type=str, default="logs_final_run")
parser.add_argument("--save_as", required=False, type=str, default="")
parser.add_argument("--num_repeats", required=False, type=int, default=2)
args = parser.parse_args()
print(args)

(
    X_train_dataloader,
    Y_train_dataloaders,
    X_test_dataloader,
    Y_test_dataloaders,
) = get_data(filename="./datasets/hvg.h5ad")

noise_dim = args.noise_dim
num_repeats = args.num_repeats


def generate_perturbations(X_test_dataloader, pi_net, inv_pcamatrix):
    # Impute samples and concat them into a dataloader
    gen_samples = {
        0: torch.Tensor([]),
        10: torch.Tensor([]),
        100: torch.Tensor([]),
        1000: torch.Tensor([]),
        10000: torch.Tensor([]),
    }

    test_source = X_test_dataloader
    for idx, batch_ in enumerate(test_source):
        batch = batch_[0]
        n_x = len(batch)
        # transport these cells below
        # and store in a sample
        gen_samples[0] = torch.cat((gen_samples[0], batch), 0)

        gen_batch_10 = transport_cot(pi_net, batch, 10)
        gen_samples[10] = torch.cat((gen_samples[10], gen_batch_10), 0)

        gen_batch_100 = transport_cot(pi_net, batch, 100)
        gen_samples[100] = torch.cat((gen_samples[100], gen_batch_100), 0)

        gen_batch_1000 = transport_cot(pi_net, batch, 1000)
        gen_samples[1000] = torch.cat((gen_samples[1000], gen_batch_1000), 0)

        gen_batch_10000 = transport_cot(pi_net, batch, 10000)
        gen_samples[10000] = torch.cat((gen_samples[10000], gen_batch_10000), 0)

    for each in gen_samples.keys():
        gen_samples[each] = gen_samples[each].detach().cpu() @ torch.Tensor(
            inv_pcamatrix
        )
        # print(gen_samples[each].shape)

    return gen_samples


def transport_cot(pi_net, batch, dose):
    n_x = len(batch)
    n_noise = 1
    noise_dim = 5
    ZX = torch.cat([batch, torch.Tensor([dose]).repeat(n_x, 1)], dim=1).to(device)
    n_s = torch.randn(n_noise, noise_dim).to(device)
    ZXn = torch.cat([ZX.repeat_interleave(n_noise, 0), n_s.repeat(n_x, 1)], dim=1).to(
        device
    )
    with torch.no_grad():
        gen_batch = pi_net(ZXn).detach().cpu()

    # print("print shape")
    # print(gen_batch.shape)
    return gen_batch


def get_test_samples(Y_test_dataloaders, inv_pcamatrix):
    dosages = [10, 100, 1000, 10000]
    test_samples = {
        10: torch.Tensor([]),
        100: torch.Tensor([]),
        1000: torch.Tensor([]),
        10000: torch.Tensor([]),
    }
    for dose in dosages:
        for idx, batch in enumerate(Y_test_dataloaders[int(f"{dose}")]):
            # print(test_samples)
            print(idx)
            if test_samples[dose].nelement() != 0:
                test_samples[dose] = torch.cat((test_samples[dose], batch[0]), 0)
            else:
                test_samples[dose] = batch[0]
            print(test_samples[dose].shape)
        test_samples[dose] = test_samples[dose] @ inv_pcamatrix

    return test_samples


def compute_ps(gen_samples, test_samples, marker_indices):
    dose_ps = []
    drug_ps = []
    ### Per dose marker genes
    for each_key in test_samples.keys():
        if each_key == 0:
            continue
        gen_mean_vec = torch.mean(gen_samples[each_key], dim=0)
        y_mean_vec = torch.mean(test_samples[each_key], dim=0)
        l2_ps = torch.linalg.norm(
            gen_mean_vec[marker_indices[f"{each_key}"]]
            - y_mean_vec[marker_indices[f"{each_key}"]]
        )
        print(f"{each_key}: {l2_ps}")
        dose_ps.append(l2_ps)

    ### per_drug marker_gene
    for each_key in test_samples.keys():
        # print(each_key)
        gen_mean_vec = torch.mean(gen_samples[each_key], dim=0)
        y_mean_vec = torch.mean(test_samples[each_key], axis=0)
        # print(test_samples[each_key].shape)
        l2_ps = torch.linalg.norm(
            gen_mean_vec[marker_indices[f"0"]] - y_mean_vec[marker_indices[f"0"]]
        )
        print(f"{each_key}: {l2_ps}")
        drug_ps.append(l2_ps)

    return np.array(dose_ps), np.array(drug_ps)


def compute_mmd(gen_samples, test_samples, marker_indices):
    dose_mmd = []
    drug_mmd = []
    for each_key in test_samples.keys():
        if each_key == 0:
            continue
        mmd_dist = compute_scalar_mmd(
            gen_samples[each_key][:, marker_indices[f"{each_key}"]].detach().cpu(),
            test_samples[each_key][:, marker_indices[f"{each_key}"]].detach().cpu(),
        )
        print(f"{each_key}: {mmd_dist}")
        dose_mmd.append(mmd_dist)

    for each_key in test_samples.keys():
        mmd_dist = compute_scalar_mmd(
            gen_samples[each_key][:, marker_indices[f"0"]].detach().cpu(),
            test_samples[each_key][:, marker_indices[f"0"]].detach().cpu(),
        )
        print(f"{each_key}: {mmd_dist}")
        drug_mmd.append(mmd_dist)

    return dose_mmd, drug_mmd


def compute_wdist(gen_samples, test_samples, marker_indices):
    dose_wdist = []
    drug_wdist = []
    for each_key in test_samples.keys():
        if each_key == 0:
            continue
        wdist = compute_wasserstein_loss(
            gen_samples[each_key][:, marker_indices[f"{each_key}"]].detach().cpu().numpy(),
            test_samples[each_key][:, marker_indices[f"{each_key}"]].detach().cpu().numpy(),
        )
        print(f"{each_key}: {wdist}")
        dose_wdist.append(wdist)

    for each_key in test_samples.keys():
        wdist = compute_wasserstein_loss(
            gen_samples[each_key][:, marker_indices[f"0"]].detach().cpu().numpy(),
            test_samples[each_key][:, marker_indices[f"0"]].detach().cpu().numpy(),
        )
        print(f"{each_key}: {wdist}")
        drug_wdist.append(wdist)

    return dose_wdist, drug_wdist


def main():
    pi_net = Generator(noise_dim=noise_dim)
    pi_net = pi_net.to(device)

    pi_net.load_state_dict(torch.load("model_trained_500.pth"))
    pi_net.eval()

    adata = sc.read("./datasets/hvg.h5ad")
    inv_pcamatrix = np.linalg.pinv(adata.obsm["X_pca"]) @ adata.X.A

    X_train_dl, Y_train_dl, X_test_dl, Y_test_dls = get_data(
        filename="./datasets/hvg.h5ad"
    )
    test_samples = get_test_samples(Y_test_dls, inv_pcamatrix)

    with open("marker_indices.json") as infile:
        marker_indices = json.load(infile)

    ps_dict = {"drug": np.zeros((num_repeats,4)), "dose":np.zeros((num_repeats, 4))}
    mmd_dict = {"drug": np.zeros((num_repeats,4)), "dose":np.zeros((num_repeats, 4))}
    ws_dict = {"drug": np.zeros((num_repeats,4)), "dose":np.zeros((num_repeats, 4))}

    for trial in tqdm(range(num_repeats)):
        # Generate Predicted samples,
        gen_samples = generate_perturbations(X_test_dl, pi_net, inv_pcamatrix)
        # compare l2
        dose_ps, drug_ps = compute_ps(gen_samples, test_samples, marker_indices)
        # compare mmd
        dose_mmd, drug_mmd = compute_mmd(gen_samples, test_samples, marker_indices)

        # compare wdist
        dose_wdist, drug_wdist = compute_wdist(
            gen_samples, test_samples, marker_indices
        )

        # append results to dict
        ps_dict['drug'][trial] = drug_ps
        ps_dict['dose'][trial] = dose_ps
        
        mmd_dict['drug'][trial] = drug_mmd
        mmd_dict['dose'][trial] = dose_mmd

        ws_dict['drug'][trial] = drug_wdist
        ws_dict['dose'][trial] = dose_wdist
    # calculate averages and print
    # TODO: also dump the json with the averages somewhere.
    print('Final Evals')
    print('PS')
    print(ps_dict['dose'].mean(axis=0))
    print(ps_dict['drug'].mean(axis=0))

    print('MMD')
    print(mmd_dict['dose'].mean(axis=0))
    print(mmd_dict['drug'].mean(axis=0))

    print('Wasserstein')
    print(ws_dict['dose'].mean(axis=0))
    print(ws_dict['drug'].mean(axis=0))

if __name__ == "__main__":
    main()
