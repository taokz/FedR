from utils_NoGE import compute_weighted_adj_matrix
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

def get_all_clients(all_data, args):
    all_rel = np.array([], dtype=int)
    for data in all_data:
        all_rel = np.union1d(all_rel, data['train']['edge_type_ori']).reshape(-1)
    nrelation = len(all_rel) # all relations of training set in all clients

    train_dataloader_list = []
    test_dataloader_list = []
    valid_dataloader_list = []
    all_dataloader_list = []

    ent_embed_list = []
    train_adj_list = []
    rel_freq_list = []

    for data in tqdm(all_data):
        nentity = len(np.unique(data['train']['edge_index'])) # entities of training in a client

        train_triples = np.stack((data['train']['edge_index'][0],
                                  data['train']['edge_type_ori'],
                                  data['train']['edge_index'][1])).T

        valid_triples = np.stack((data['valid']['edge_index'][0],
                                  data['valid']['edge_type_ori'],
                                  data['valid']['edge_index'][1])).T

        test_triples = np.stack((data['test']['edge_index'][0],
                                 data['test']['edge_type_ori'],
                                 data['test']['edge_index'][1])).T

        client_mask_rel = np.setdiff1d(np.arange(nrelation),
                                       np.unique(data['train']['edge_type_ori'].reshape(-1)), assume_unique=True)

        all_triples = np.concatenate([train_triples, valid_triples, test_triples])

        #train_valid_triples = np.concatenate([train_triples, valid_triples])
        train_adj = compute_weighted_adj_matrix(train_triples.tolist(), nrelation).to(args.gpu) #change: nentity

        train_dataloader_list.append(train_triples.tolist())
        valid_dataloader_list.append(valid_triples.tolist())
        test_dataloader_list.append(test_triples.tolist())
        all_dataloader_list.append(all_triples.tolist())
        train_adj_list.append(train_adj)

        embeddings = torch.nn.Embedding(nentity, args.emb_dim)
        ent_embed = torch.nn.init.xavier_normal_(embeddings.weight.data)
        ent_embed_list.append(ent_embed)
        
        # used for aggregation
        rel_freq = torch.zeros(nrelation)
        for r in data['train']['edge_type_ori'].reshape(-1):
            rel_freq[r] += 1
        rel_freq_list.append(rel_freq)

    rel_freq_mat = torch.stack(rel_freq_list).to(args.gpu)

    return train_dataloader_list, valid_dataloader_list, test_dataloader_list, all_dataloader_list, \
          rel_freq_mat, ent_embed_list, train_adj_list, nrelation
