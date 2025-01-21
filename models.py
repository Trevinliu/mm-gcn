import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from layers import GraphConvolution
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pdb

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, input_dim // 4)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.SELU()
        self.output_fc = nn.Linear(input_dim // 4, 1)
        self.LogSoftmax = nn.LogSoftmax()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, task):
        # Reshape input data
        batch_size = x.shape[0]
        x1 = x.view(batch_size, -1)
        l1 = self.input_fc(x1)
        l2 = self.drop(self.activation(l1))
        l2 = self.output_fc(l2)
        if task == 'sex_predict':
            pred = self.Sigmoid(l2)
        elif task == 'age_predict':
            pred = l2
        return pred.reshape(pred.shape[0])

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batch = nn.BatchNorm1d(input_dim // 2 * hidden_dim // 2)
        self.fc1 = nn.Linear(input_dim // 2 * hidden_dim // 2, 128)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, con, task):
        batch_size = con.size(0)
        con_size = con.size(1) * con.size(2)
        x = F.selu(self.gc1(x, con))
        x = self.drop(x)
        l1 = self.avgpool(x)
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.selu(self.fc1(l1))
        l2 = self.drop(l2)
        l2 = self.fc2(l2)
        if task == 'sex_predict':
            pred = self.Sigmoid(l2)
        elif task == 'age_predict':
            pred = l2
        return pred.reshape(pred.shape[0])

class GCN_specific_L_W(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN_specific_L_W, self).__init__()
        c_num = (input_dim // 2) * (input_dim // 2) * 2
        self.learn_weight_layer = nn.Sequential(
            nn.BatchNorm1d(c_num),
            nn.Linear(c_num, c_num // 20),
            nn.ReLU(),
            nn.Linear(c_num // 20, input_dim // 2)
        )
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batch = nn.BatchNorm1d(input_dim // 2 * hidden_dim // 2)
        self.fc1 = nn.Linear(input_dim // 2 * hidden_dim // 2, 128)
        self.drop = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, joint, task):
        FC = joint[:, 188:376, 188:376]
        SC = joint[:, 0:188, 0:188]
        b, w, h = FC.shape

        FC_flat = FC.reshape(b, -1)
        SC_flat = SC.reshape(b, -1)
        FC_SC_flat = torch.cat([FC_flat, SC_flat], dim=1)

        weight = self.learn_weight_layer(FC_SC_flat.float())
        weight = torch.sigmoid(weight)

        mapcat = torch.zeros_like(FC)
        for i in range(w):
            mapcat[:, i, i] = weight[:, i]

        mapcatsub1 = torch.cat((FC, mapcat), 1)
        mapcatsub2 = torch.cat((mapcat, SC), 1)
        adj = torch.cat([mapcatsub1, mapcatsub2], 2)

        x = F.selu(self.gc1(x, adj))
        x = self.drop(x)

        l1 = self.avgpool(x)
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.batch(l1)
        l2 = F.selu(self.fc1(l1))
        l2 = self.drop(l2)
        l2 = self.fc2(l2)

        if task == 'sex_predict':
            pred = self.Sigmoid(l2)
        elif task == 'age_predict':
            pred = l2
        return pred.reshape(pred.shape[0])
