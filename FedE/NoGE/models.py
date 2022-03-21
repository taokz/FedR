import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch import empty, matmul, tensor
import torch
from torch.cuda import empty_cache
from torch.nn import Parameter, Module
from torch.nn.functional import normalize
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import math
import numpy as np
from gnn_layers import *

class NoGE_GCN_QuatE(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, adj, n_entities, n_relations, num_layers=1):
        super(NoGE_GCN_QuatE, self).__init__()
        self.adj = adj
        self.num_layers = num_layers
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)

        self.lst_gcn = torch.nn.ModuleList()
        for _layer in range(self.num_layers):
            if _layer == 0:
                self.lst_gcn.append(GraphConvolution(emb_dim, hid_dim, act=torch.tanh))
            else:
                self.lst_gcn.append(GraphConvolution(hid_dim, hid_dim, act=torch.tanh))

        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        #print(1, self.adj.get_device())
        for _layer in range(self.num_layers):
            X = self.lst_gcn[_layer](X, self.adj)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        normalized_r = normalization(r)
        hr = vec_vec_wise_multiplication(h, normalized_r) # following the 1-N scoring strategy
        hr = self.hidden_dropout2(hr)
        hrt = torch.mm(hr, X[:self.n_entities].t())
        pred = torch.sigmoid(hrt)
        return pred

# Quaternion operations
def normalization(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
    size = quaternion.size(split_dim) // 4
    quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
    quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # quaternion / norm
    quaternion = quaternion.reshape(-1, 4 * size)
    return quaternion

def make_wise_quaternion(quaternion):  # for vector * vector quaternion element-wise multiplication
    if len(quaternion.size()) == 1:
        quaternion = quaternion.unsqueeze(0)
    size = quaternion.size(1) // 4
    r, i, j, k = torch.split(quaternion, size, dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3 --> bs x 4dim
    i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
    return r2, i2, j2, k2

def get_quaternion_wise_mul(quaternion):
    size = quaternion.size(1) // 4
    quaternion = quaternion.view(-1, 4, size)
    quaternion = torch.sum(quaternion, 1)
    return quaternion

def vec_vec_wise_multiplication(q, p):  # vector * vector
    q_r, q_i, q_j, q_k = make_wise_quaternion(q)  # bs x 4dim

    qp_r = get_quaternion_wise_mul(q_r * p)  # qrpr−qipi−qjpj−qkpk
    qp_i = get_quaternion_wise_mul(q_i * p)  # qipr+qrpi−qkpj+qjpk
    qp_j = get_quaternion_wise_mul(q_j * p)  # qjpr+qkpi+qrpj−qipk
    qp_k = get_quaternion_wise_mul(q_k * p)  # qkpr−qjpi+qipj+qrpk

    return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)

# def regularization(quaternion):  # vectorized quaternion bs x 4dim
#     size = quaternion.size(1) // 4
#     r, i, j, k = torch.split(quaternion, size, dim=1)
#     return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)