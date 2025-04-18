"""
This file contains the utility functions for the data provider.
Adapted from: https://github.com/dwromero/ckconv/blob/master/datasets/utils.py
"""
import torch
import sklearn.model_selection
import os


def normalise_data_alternative(tensors, train_tensor):
    """Normalize multiple tensors using statistics from train_tensor."""
    out = []
    for Xi in tensors:
        normalized = []
        for dim, train_dim in zip(Xi.unbind(dim=-1), train_tensor.unbind(dim=-1)):
            train_nonan = train_dim.masked_select(~torch.isnan(train_dim))
            mean = train_nonan.mean() # compute statistics using only training data.
            std = train_nonan.std() # compute statistics using only training data.
            normalized.append((dim - mean) / (std + 1e-5))
        out.append(torch.stack(normalized, dim=-1))
    return out

def split_train_val(tensor_X, tensor_y):
    """Split training data into train and validation sets."""
    train_X, val_X, train_y, val_y = sklearn.model_selection.train_test_split(
        tensor_X, tensor_y,
        train_size=0.85,
        random_state=0,
        shuffle=True,
        stratify=tensor_y
    )
    return (train_X, val_X), (train_y, val_y)



def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out

def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (
        train_tensor,
        testval_tensor,
        train_stratify,
        testval_stratify,
    ) = sklearn.model_selection.train_test_split(
        tensor,
        stratify,
        train_size=0.7,
        random_state=0,
        shuffle=True,
        stratify=stratify,
    )

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(
        testval_tensor,
        train_size=0.5,
        random_state=1,
        shuffle=True,
        stratify=testval_stratify,
    )
    return train_tensor, val_tensor, test_tensor

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + ".pt")

def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors