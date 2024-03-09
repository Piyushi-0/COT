# Imports
import argparse
import os

import anndata
import numpy as np
import scanpy as sc
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms



def read_data(filename="hvg.h5ad"):
    adata = sc.read(filename)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True)

    dat = adata.obsm["X_pca"]

    # Separate out control cells,
    ctrl_indices = np.where(adata.obs["dose"].to_numpy() == 0)[0]
    print(f"Number of control cells:{len(ctrl_indices)}")
    ctrl_data = dat[ctrl_indices]

    # Different y distributions
    y_1 = dat[np.where(adata.obs["drug-dose"].to_numpy() == "givinostat-10")]
    y_2 = dat[np.where(adata.obs["drug-dose"].to_numpy() == "givinostat-100")]
    y_3 = dat[np.where(adata.obs["drug-dose"].to_numpy() == "givinostat-1000")]
    y_4 = dat[np.where(adata.obs["drug-dose"].to_numpy() == "givinostat-10000")]

    return ctrl_data, y_1, y_2, y_3, y_4


def read_data_unstratify(filename='hvg.h5ad'):
    """

    """
    adata = sc.read(filename)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True)

    dat = adata.obsm["X_pca"]

    # Separate out control cells,
    ctrl_indices = np.where(adata.obs["dose"].to_numpy() == 0)[0]
    print(f"Number of control cells:{len(ctrl_indices)}")
    ctrl_data = dat[ctrl_indices]

    # Different y distributions
    y_indices = np.where(adata.obs["drug"].to_numpy() == "givinostat")[0]


    train_set = []
    test_set = []

    ctrl_train, ctrl_test = train_test_split(ctrl_data,test_size=0.2, random_state=0)
    train_set.append(ctrl_train)
    test_set.append(ctrl_test)

    y_train, y_test = train_test_split(y_indices, test_size=0.2, random_state=0)
    adata.obs['split'] = 'ignore'
    adata.obs['split'].iloc[y_train] = 'train'
    adata.obs['split'].iloc[y_test] = 'test'

    print(adata.obs['split'].unique())

    train_y_1 = dat[np.where((adata.obs["drug-dose"] == "givinostat-10") & (adata.obs["split"] == 'train'))]
    train_y_2 = dat[np.where((adata.obs["drug-dose"] == "givinostat-100") & (adata.obs["split"] == 'train'))]
    train_y_3 = dat[np.where((adata.obs["drug-dose"] == "givinostat-1000") & (adata.obs["split"] == 'train'))]
    train_y_4 = dat[np.where((adata.obs["drug-dose"] == "givinostat-10000") & (adata.obs["split"] == 'train'))]

    train_set.append(train_y_1)
    train_set.append(train_y_2)
    train_set.append(train_y_3)
    train_set.append(train_y_4)

    y_1 = dat[np.where((adata.obs["drug-dose"] == "givinostat-10") & (adata.obs["split"] == 'test'))]
    y_2 = dat[np.where((adata.obs["drug-dose"] == "givinostat-100") & (adata.obs["split"] == 'test'))]
    y_3 = dat[np.where((adata.obs["drug-dose"] == "givinostat-1000") & (adata.obs["split"].to_numpy() == 'test'))]
    y_4 = dat[np.where((adata.obs["drug-dose"] == "givinostat-10000") &(adata.obs["split"].to_numpy() == 'test'))]
    print(y_1.shape)

    test_set.append(y_1)
    test_set.append(y_2)
    test_set.append(y_3)
    test_set.append(y_4)

    return train_set, test_set

def split_data(X, Y1, Y2, Y3, Y4):
    """
    Split data into training and test datasets
    """
    train_set = []
    test_set = []
    for each in (X, Y1, Y2, Y3, Y4):
        each_train, each_test = train_test_split(each, test_size=0.2, random_state=0)
        train_set.append(each_train)
        test_set.append(each_test)

    return tuple(train_set), tuple(test_set)


def get_dataloaders(dataset, batch_size=512, holdout=None):
    """
    Dataset here is not a pytorch dataset, but rather a tuple
    """
    tensor_dl = {}

    source_dset = TensorDataset(torch.from_numpy(dataset[0]).float())
    source_dl = DataLoader(source_dset, batch_size=batch_size, shuffle=True)

    doses = [10, 100, 1000, 10000]
    for i, each in enumerate(dataset):
        if i == 0:
            continue
        each_dset = TensorDataset(torch.from_numpy(each).float())
        print(torch.from_numpy(each).shape)

        dl = DataLoader(each_dset, batch_size=batch_size, shuffle=True)
        tensor_dl[doses[i - 1]] = dl

    if holdout:
        tensor_dl.pop(holdout, None)

    return source_dl, tensor_dl


def get_data(filename="hvg.h5ad", holdout=None, holdout_test=False):
    """
    Return train and test dataloaders
    """
    #X, y_1, y_2, y_3, y_4 = read_data(filename)
    train_set, test_set = read_data_unstratify(filename)
    #train_set, test_set = split_data(X, y_1, y_2, y_3, y_4)
    X_train_dataloader, Y_train_dataloaders = get_dataloaders(train_set, holdout=holdout)
    if holdout_test:
        X_test_dataloader, Y_test_dataloaders = get_dataloaders(test_set, holdout=holdout)
    else:
        X_test_dataloader, Y_test_dataloaders = get_dataloaders(test_set)


    return (
        X_train_dataloader,
        Y_train_dataloaders,
        X_test_dataloader,
        Y_test_dataloaders,
    )
