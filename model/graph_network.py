import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.file_and_ckpt import read_prior


class GCN(nn.Module):
    def __init__(self, config, label_ids):
        super(GCN, self).__init__()
        self.config = config
        self.in_dim = self.config.model.structure_encoder.dimension
        self.weight = nn.Parameter(torch.FloatTensor(self.in_dim, self.in_dim))
        self.bias = nn.Parameter(torch.zeros((1, 1, self.in_dim), dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.reset_parameters()

        topdown_prior, bottomup_prior = read_prior(self.config, label_ids)
        loop = np.eye(len(label_ids), dtype=np.float32)
        A = (topdown_prior + bottomup_prior + loop) > 0.
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32, requires_grad=False))
        D_inv = torch.zeros_like(self.A)
        for i in range(self.A.size(0)):
            D_inv[i, i] = 1. / torch.sum(self.A[i, :])
        self.A = torch.matmul(D_inv, self.A)

    def reset_parameters(self):
        self.weight.data.uniform_(-1. / math.sqrt(self.weight.size(1)), 1. / math.sqrt(self.weight.size(1)))

    def forward(self, inputs):
        x = torch.einsum('bvi,ij->bvj', inputs, self.weight)
        x = torch.einsum('ij,bjd->bid', self.A, x)
        return x + self.bias


class HierarchyGCN(nn.Module):
    def __init__(self, config, label_ids):
        super(HierarchyGCN, self).__init__()
        self.config = config
        topdown_prior, bottomup_prior = read_prior(self.config, label_ids)
        self.register_buffer('topdown_prior', torch.tensor(topdown_prior, dtype=torch.float32, requires_grad=False))
        self.register_buffer('bottomup_prior', torch.tensor(bottomup_prior, dtype=torch.float32, requires_grad=False))

        self.in_dim = self.config.model.structure_encoder.dimension
        # TOPDOWN GCN
        self.topdown_bias1 = nn.Parameter(torch.zeros([1, len(label_ids), self.in_dim], dtype=torch.float32))
        self.topdown_bias2 = nn.Parameter(torch.zeros([1, len(label_ids), 1], dtype=torch.float32))
        self.topdown_fc = nn.Linear(self.in_dim, 1, bias=False)
        # BOTTOMUP GCN
        self.bottomup_bias1 = nn.Parameter(torch.zeros([1, len(label_ids), self.in_dim], dtype=torch.float32))
        self.bottomup_bias2 = nn.Parameter(torch.zeros([1, len(label_ids), 1], dtype=torch.float32))
        self.bottomup_fc = nn.Linear(self.in_dim, 1, bias=False)
        # LOOP CONNECTION GCN
        self.loop_fc = nn.Linear(self.in_dim, 1, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.model.structure_encoder.dropout)
    
    def forward(self, inputs):
        topdown_gate = self.topdown_fc(inputs + self.topdown_bias2)
        bottomup_gate = self.bottomup_fc(inputs + self.bottomup_bias2)
        loop_gate = self.loop_fc(inputs)

        topdown_message = (torch.matmul(self.topdown_prior, inputs) + self.topdown_bias1) * F.sigmoid(topdown_gate)
        bottomup_message = (torch.matmul(self.bottomup_prior, inputs) + self.bottomup_bias1) * F.sigmoid(bottomup_gate)
        loop_message = inputs * F.sigmoid(loop_gate)
        
        return self.relu(self.dropout(topdown_message) + self.dropout(bottomup_message) + self.dropout(loop_message))
        

class DotProductAttentionGCN(nn.Module):
    def __init__(self, config, label_ids):
        super(DotProductAttentionGCN, self).__init__()
        self.config = config
        topdown_prior, bottomup_prior = read_prior(self.config, label_ids)
        loop = np.eye(len(label_ids), dtype=np.float32)
        mask = (topdown_prior + bottomup_prior + loop) > 0.
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32, requires_grad=False).unsqueeze(0))

        self.in_dim = self.config.model.structure_encoder.dimension
        self.q = nn.Linear(self.in_dim, self.in_dim)
        self.k = nn.Linear(self.in_dim, self.in_dim)
        self.v = nn.Linear(self.in_dim, self.in_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.model.structure_encoder.dropout)
    
    def forward(self, inputs):
        query = self.q(inputs)
        key = self.k(inputs)
        value = self.v(inputs)
        dot_product = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.in_dim)
        masked = torch.exp(dot_product) * self.mask
        attention = masked / torch.sum(masked, dim=2, keepdim=True)
        outputs = torch.bmm(attention, value)

        return self.relu(self.dropout(outputs))


class GraphAttentionNetwork(nn.Module):
    def __init__(self, config, label_ids):
        super(GraphAttentionNetwork, self).__init__()
        self.config = config
        topdown_prior, bottomup_prior = read_prior(self.config, label_ids)
        loop = np.eye(len(label_ids), dtype=np.float32)
        mask = (topdown_prior + bottomup_prior + loop) > 0.
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32, requires_grad=False).unsqueeze(0))

        self.in_dim = self.config.model.structure_encoder.dimension

        self.W = nn.Linear(self.in_dim, self.in_dim)
        self.Aq = nn.Linear(self.in_dim, 1, bias=False)
        self.Ak = nn.Linear(self.in_dim, 1, bias=False)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.model.structure_encoder.dropout)

    def forward(self, inputs):
        feature = self.W(inputs)
        query = self.Aq(feature)
        key = self.Ak(feature)
        coef = self.lrelu(query + key.transpose(1, 2))
        masked = torch.exp(coef) * self.mask
        attention = masked / torch.sum(masked, dim=2, keepdim=True)
        outputs = torch.bmm(attention, feature)
            
        return self.relu(self.dropout(outputs))


class DotProductGraphAttentionNetwork(nn.Module):
    def __init__(self, config, label_ids):
        super(DotProductGraphAttentionNetwork, self).__init__()
        self.config = config
        topdown_prior, bottomup_prior = read_prior(self.config, label_ids)
        loop = np.eye(len(label_ids), dtype=np.float32)
        mask = (topdown_prior + bottomup_prior + loop) > 0.
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32, requires_grad=False).unsqueeze(0))

        self.in_dim = self.config.model.structure_encoder.dimension

        self.W = nn.Linear(self.in_dim, self.in_dim)
        self.Aq = nn.Linear(self.in_dim, self.config.model.gcn_setting.dim, bias=False)
        self.Ak = nn.Linear(self.in_dim, self.config.model.gcn_setting.dim, bias=False)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.model.structure_encoder.dropout)

    def forward(self, inputs):
        feature = self.W(inputs)
        query = self.Aq(feature)
        key = self.Ak(feature)
        coef = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.config.model.gcn_setting.dim)
        masked = torch.exp(coef) * self.mask
        attention = masked / torch.sum(masked, dim=2, keepdim=True)
        outputs= torch.bmm(attention, feature)
        
        return self.relu(self.dropout(outputs))
