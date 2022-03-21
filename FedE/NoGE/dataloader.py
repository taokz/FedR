from utils_NoGE import compute_weighted_adj_matrix
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

def get_all_clients(all_data, args):
    all_ent = np.array([], dtype=int)
    for data in all_data:
        all_ent = np.union1d(all_ent, data['train']['edge_index_ori'].reshape(-1))
    nentity = len(all_ent)
    if 'DDB14' in args.data_path:
        nentity = 9203

    train_dataloader_list = []
    test_dataloader_list = []
    valid_dataloader_list = []
    all_dataloader_list = []
    #rel_num_list = []
    rel_embed_list = []
    train_adj_list = []

    ent_freq_list = []

    for data in tqdm(all_data):
        nrelation = len(np.unique(data['train']['edge_type']))

        train_triples = np.stack((data['train']['edge_index_ori'][0],
                                  data['train']['edge_type'],
                                  data['train']['edge_index_ori'][1])).T

        valid_triples = np.stack((data['valid']['edge_index_ori'][0],
                                  data['valid']['edge_type'],
                                  data['valid']['edge_index_ori'][1])).T

        test_triples = np.stack((data['test']['edge_index_ori'][0],
                                 data['test']['edge_type'],
                                 data['test']['edge_index_ori'][1])).T

        client_mask_ent = np.setdiff1d(np.arange(nentity),
                                       np.unique(data['train']['edge_index_ori'].reshape(-1)), assume_unique=True)

        all_triples = np.concatenate([train_triples, valid_triples, test_triples])

        train_adj = compute_weighted_adj_matrix(train_triples.tolist(), nentity).to(args.gpu)

        train_dataloader_list.append(train_triples.tolist())
        valid_dataloader_list.append(valid_triples.tolist())
        test_dataloader_list.append(test_triples.tolist())
        all_dataloader_list.append(all_triples.tolist())
        #rel_num_list.append(nrelation)
        train_adj_list.append(train_adj)

        embeddings = torch.nn.Embedding(nrelation, args.emb_dim)
        rel_embed = torch.nn.init.xavier_normal_(embeddings.weight.data)
        rel_embed_list.append(rel_embed)
        
        # used for aggregation
        ent_freq = torch.zeros(nentity)
        for e in data['train']['edge_index_ori'].reshape(-1):
            ent_freq[e] += 1
        ent_freq_list.append(ent_freq)

    ent_freq_mat = torch.stack(ent_freq_list).to(args.gpu)

    return train_dataloader_list, valid_dataloader_list, test_dataloader_list, all_dataloader_list, \
          ent_freq_mat, rel_embed_list, train_adj_list, nentity
