import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats
import config
import pdb
import logging

# Compute node features. The first 182 dimensions store part of the connectivity information, while the last 6 dimensions store statistical features related to node degree.
def compute_degree_features(connectivity_matrices):
    num_samples, num_nodes = connectivity_matrices.shape[0], connectivity_matrices.shape[1]
    # Assume we keep the first 182 connectivity features and 6 degree features
    features = torch.zeros(num_samples, num_nodes, 188)

    for sample_idx in range(num_samples):
        for node_idx in range(num_nodes):
            # Directly use part of the connectivity information
            features[sample_idx, node_idx, :182] = connectivity_matrices[sample_idx, node_idx, :182]

            # Calculate and store degree features
            degree = connectivity_matrices[sample_idx, node_idx, :].sum()
            neighbors = (connectivity_matrices[sample_idx, node_idx, :] > 0).nonzero(as_tuple=True)[0]
            neighbor_degrees = degree if neighbors.nelement() == 0 else connectivity_matrices[sample_idx, neighbors, :].sum(axis=1)

            features[sample_idx, node_idx, 182] = degree  # Node's own degree
            features[sample_idx, node_idx, 183] = neighbor_degrees.min() if neighbors.nelement() > 0 else 0  # Minimum degree
            features[sample_idx, node_idx, 184] = neighbor_degrees.max() if neighbors.nelement() > 0 else 0  # Maximum degree
            features[sample_idx, node_idx, 185] = neighbor_degrees.mean() if neighbors.nelement() > 0 else 0  # Mean degree
            features[sample_idx, node_idx, 186] = neighbor_degrees.std() if neighbors.nelement() > 0 else 0  # Degree standard deviation

    return features

def generator(train_idx, test_idx, batch_size, task, FC_features, SC_features):

    label = pd.read_csv(config.label_file)[task.split('_')[0]].values
    # Assume label_encoded is a numpy array or similar list containing labels for 196 samples
    labels_tensor = torch.FloatTensor(label)

    # Directly convert FC and SC data to PyTorch Tensors
    FC_tensor = torch.stack([torch.tensor(fc, dtype=torch.float32) for fc in FC_features])
    SC_tensor = torch.stack([torch.tensor(sc, dtype=torch.float32) for sc in SC_features])

    # Node features generated from FC
    features_tensor = compute_degree_features(FC_tensor)

    # Divide training data
    train_FC = FC_tensor[train_idx]
    train_SC = SC_tensor[train_idx]
    train_features = features_tensor[train_idx]
    train_labels = labels_tensor[train_idx]

    # Generate training data
    train_set = TensorDataset(train_FC, train_SC, train_features, train_labels)
    train_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)

    # Divide testing data
    test_FC = FC_tensor[test_idx]
    test_SC = SC_tensor[test_idx]
    test_features = features_tensor[test_idx]
    test_labels = labels_tensor[test_idx]

    # Generate testing data
    test_set = TensorDataset(test_FC, test_SC, test_features, test_labels)
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_generator, test_generator

def accuracy(output, labels):
    y_pred = torch.round(output)
    correct = y_pred.eq(labels.view_as(y_pred)).sum()
    acc = correct.float() / labels.shape[0]
    return acc

if __name__ == '__main__':
    f_list = np.load(config.Concat_Feature_path, allow_pickle=True)
    print(f_list.shape)
