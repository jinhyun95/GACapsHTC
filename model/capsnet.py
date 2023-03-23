import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.file_and_ckpt import make_label_indices, read_hierarchy


class DALayer(nn.Module):
    # Dual Attention Capsule Network
    def __init__(self, num_caps, in_dim):
        super(DALayer, self).__init__()
        self.layer1 = nn.Linear(in_dim * num_caps, num_caps)
        self.layer2 = nn.Linear(num_caps, num_caps)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x, additional=None):
        if additional is not None:
            flat = torch.cat([x, additional], dim=2).reshape(x.size(0), -1)
        else:
            flat = x.reshape(x.size(0), -1) # B VD
        attention = self.tanh(self.layer2(self.relu(self.layer1(flat)))).unsqueeze(-1)
        x = x * (1. + attention)

        return x

class CapsuleLayer(nn.Module):
    def __init__(self, config, label_ids):
        super(CapsuleLayer, self).__init__()
        self.config = config
        self.num_caps = len(label_ids)
        self.W = nn.Linear(in_features=self.config.model.structure_encoder.dimension,
                            out_features=self.config.model.capsule_setting.dimension * self.num_caps,
                            bias=False)
        nn.init.uniform_(self.W.weight.data, - 1. / math.sqrt(self.num_caps), 1. / math.sqrt(self.num_caps))
        bias = torch.zeros([1, self.num_caps, self.config.model.capsule_setting.dimension * self.num_caps],
                            dtype=torch.float32, requires_grad=True)
        self.bias = nn.Parameter(bias)
        self.scaler = 1.
        self.dropout = self.config.model.capsule_setting.dropout
        self.attention = DALayer(self.num_caps, self.config.model.structure_encoder.dimension) if self.config.model.capsule_setting.attention \
                         else nn.Identity()

    def squash(self, x, dim):
        norm = torch.norm(x, dim=dim, keepdim=True)
        s = norm / (0.5 + torch.pow(norm, 2)) * x
        return s

    def routing(self, coef, u, num_iter):
        for _ in range(num_iter):
            beta = F.softmax(coef, dim=2)
            s = torch.sum(beta * u, dim=1, keepdim=True)
            cc = self.squash(s, dim=3)
            coef = coef + torch.sum(u * cc, dim=3, keepdim=True)
        return coef
    
    def masked_routing(self, coef, u, num_iter, mask):
        for _ in range(num_iter):
            beta = masked_softmax(mask, coef, dim=2)
            s = torch.sum(beta * u, dim=1, keepdim=True)
            cc = self.squash(s, dim=3)
            coef = coef + torch.sum(u * cc, dim=3, keepdim=True)
        return coef

    def forward(self, x):
        # x: BVD
        x = self.attention(x)
        p = self.W(x) + self.bias
        p = p.view(x.size(0), self.num_caps, self.num_caps, self.config.model.capsule_setting.dimension) * self.scaler
        self.p = p.detach()
        if self.training and self.dropout > 0:
            dropout_mask = (torch.rand([x.size(0), x.size(1), 1, 1]) > self.dropout).type(torch.float32).to(x.device)
            p = p * dropout_mask / math.sqrt(1. - self.dropout)
            
        b = torch.zeros(x.size(0), self.num_caps, self.num_caps, 1, requires_grad=False).to(x.device)
        # DETACHED ROUTING PROCESS
        # CVPR 20, IMPROVING THE ROBUSTNESS OF CAPSULE NETWORKS TO IMAGE AFFINE TRANSFORMATION
        with torch.no_grad():
            b = self.routing(b, p.detach(), self.config.model.capsule_setting.iter - 1)
            beta = F.softmax(b, dim=2)
        s = torch.sum(beta * p, dim=1)
        cc = self.squash(s, dim=2)

        return torch.clamp(torch.norm(cc, dim=2), min=1e-6, max=1.-1e-6), p


class KDECapsuleLayer(nn.Module):
    # ZHAO 2019, Towards Scalable and Reliable Capsule Networks for Challenging NLP Applications
    def __init__(self, config, label_ids):
        super(KDECapsuleLayer, self).__init__()
        self.config = config
        self.num_caps = len(label_ids)
        self.W = nn.Linear(in_features=self.config.model.structure_encoder.dimension,
                           out_features=self.config.model.capsule_setting.dimension * self.num_caps,
                           bias=False)
        nn.init.uniform_(self.W.weight.data, - 1. / math.sqrt(self.num_caps), 1. / math.sqrt(self.num_caps))
        bias = torch.zeros([1, self.num_caps, self.config.model.capsule_setting.dimension * self.num_caps],
                        dtype=torch.float32, requires_grad=True)
        self.bias = nn.Parameter(bias)
        self.scaler = 1.
        self.dropout = self.config.model.capsule_setting.dropout
        self.attention = DALayer(self.num_caps, self.config.model.structure_encoder.dimension) if self.config.model.capsule_setting.attention \
                         else nn.Identity()

        label_ids = make_label_indices(config)
        self.hierarchy = read_hierarchy(config, label_ids)
        label_ids = list(label_ids.values())
        for parent in self.hierarchy.keys():
            for child in self.hierarchy[parent]:
                if child in label_ids:
                    label_ids.remove(child)
        self.hierarchy[-1] = label_ids

        self.npys = []
        self.num_instances = []
                
    def squash(self, x, dim):
        norm = torch.norm(x, dim=dim, keepdim=True)
        s = norm / (0.5 + torch.pow(norm, 2)) * x
        return s

    def routing(self, coef, u, mask=None):
        last_loss = 0.
        self.num_iter = 0
        while True:
            self.num_iter += 1
            if mask is None:
                c = F.softmax(coef, dim=2)[:, :, :-1, :]
            else:
                c = masked_softmax(mask, coef, dim=2)[:, :, :-1, :]
            
            v = self.squash((u * c).sum(dim=1, keepdim=True), dim=3)
            d = 1. - torch.norm(self.squash(u, dim=3) - v, dim=3, keepdim=True)
            coef = coef + torch.cat([d, torch.zeros_like(d[:, :, :1, :])], dim=2)
            kde = math.log((coef[:, :, :-1, :] * d).sum().item() / coef.size(0))
            if -0.05 < kde - last_loss < 0.05:
                break
            else:
                last_loss = kde
        c = F.softmax(coef, dim=2)[:, :, :-1, :]
        
        return torch.norm(v.squeeze(1), dim=2)

    def forward(self, x):
        x = self.attention(x)

        p = self.W(x) + self.bias
        p = p.view(x.size(0), self.num_caps, self.num_caps, self.config.model.capsule_setting.dimension) * self.scaler
        self.p = p.detach()

        if hasattr(self.config.model.capsule_setting, 'prune'):
            norms = torch.norm(x, dim=2)
            k = int(x.size(1) * (1. - self.config.model.capsule_setting.prune))
            pruning_mask = torch.zeros_like(norms)
            topk = torch.topk(norms, k, dim=1)[1]
            for bid in range(x.size(0)):
                pruning_mask[bid, topk[bid, :]] = 1.
            p = p * pruning_mask.unsqueeze(-1).unsqueeze(-1)
            
        if self.training and self.dropout > 0:
            dropout_mask = (torch.rand([x.size(0), x.size(1), 1, 1]) > self.dropout).type(torch.float32).to(x.device)
            p = p * dropout_mask / math.sqrt(1. - self.dropout)
        
        b = torch.zeros(x.size(0), self.num_caps, self.num_caps + 1, 1).to(x.device)
        
        activations = self.routing(b, p)

        return torch.clamp(activations, min=1e-6, max=1.-1e-6), p

def masked_softmax(mask, v, dim):
    x = torch.exp(v) * mask
    return x / x.sum(dim=dim, keepdim=True)
