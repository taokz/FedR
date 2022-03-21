import torch
import os
import numpy as np
import torch
import tqdm
import pandas as pd

# input is not a raw file --> list of clients' data 
# [client1, client2, client3] -> clientx {train, valid, test} -> train {node, node_ori, type, type_ori}
# this function is mainly used to build adjacency matrix

def load_data(data_dict, is_unweigted=False, directed=True):
    size = len(data_dict['edge_type'])
    triples_data = []
    rows, cols, data = [], [], []
    unique_entities = set()

    for i in range(size):
        e1 = data_dict['edge_index_ori'][0][i]
        e2 = data_dict['edge_index_ori'][1][i]
        r  = data_dict['edge_type'][i]

        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append((e1, r, e2))

        if not directed:
            # connecting source and tail entity
            rows.append(e1)
            cols.append(e2)
            # relation weight in adjacency matrix
            if is_unweigted:
                data.append(1)
            else:
                data.append(r)

        # connecting tail and source entity
        rows.append(e2)
        cols.append(e1)
        if is_unweigted:
            data.append(1)
        else:
            data.append(r)

    return triples_data, (rows, cols, data), list(unique_entities)


def get_all_clients(all_data, args, is_unweigted=False, directed=True):
    
    all_ent = np.array([], dtype=int)
    for data in all_data:
        all_ent = np.union1d(all_ent, data['train']['edge_index_ori']).reshape(-1)
    nentity = len(all_ent) # all entities of training set in all clients
    if 'DDB14' in args.data_path:
        nentity = 9203

    train_dataloader_list = []
    test_dataloader_list = []
    valid_dataloader_list = []
    headTailSelector_list = []
    unique_entities_train_list = []

    rel_embed_list = []
    ent_freq_list = []
   
    for data in all_data:
        nrelation = np.unique(data['train']['edge_type']) # unique relations in training set in a client
        size = len(data['train']['edge_type'])

        train_triples, train_adjacency_mat, unique_entities_train, = load_data(data['train'], is_unweigted, directed)
        validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(data['valid'], is_unweigted, directed)
        test_triples, test_adjacency_mat, unique_entities_test = load_data(data['test'], is_unweigted, directed)

        left_entity, right_entity = {}, {}

        for i in range(size):
            e1 = data['train']['edge_index_ori'][0][i]
            e2 = data['train']['edge_index_ori'][1][i]
            relation = data['train']['edge_type'][i]

            # Count number of occurences for each (e1, relation)
            if relation not in left_entity:
                left_entity[relation] = {}
            if e1 not in left_entity[relation]:
                left_entity[relation][e1] = 0
            left_entity[relation][e1] += 1

            # Count number of occurences for each (relation, e2)
            if relation not in right_entity:
                right_entity[relation] = {}
            if e2 not in right_entity[relation]:
                right_entity[relation][e2] = 0
            right_entity[relation][e2] += 1

        left_entity_avg = {}
        for j in nrelation:
            left_entity_avg[j] = sum(
                left_entity[j].values()) * 1.0 / len(left_entity[j])

        right_entity_avg = {}
        for j in nrelation:
            right_entity_avg[j] = sum(
                right_entity[j].values()) * 1.0 / len(right_entity[j])

        headTailSelector = {}
        for j in nrelation:
            headTailSelector[j] = 1000 * right_entity_avg[j] / \
                (right_entity_avg[j] + left_entity_avg[j])

        ent_freq = torch.zeros(nentity)
        for e in data['train']['edge_index_ori'].reshape(-1):
            ent_freq[e] += 1
        ent_freq_list.append(ent_freq)

        rel_embed = np.random.randn(len(nrelation), args.embedding_size)
        rel_embed_list.append(rel_embed)

        train_dataloader_list.append((train_triples, train_adjacency_mat))
        valid_dataloader_list.append((validation_triples, valid_adjacency_mat))
        test_dataloader_list.append((test_triples, test_adjacency_mat))
        headTailSelector_list.append(headTailSelector)
        unique_entities_train_list.append(unique_entities_train)

    ent_freq_mat = torch.stack(ent_freq_list).cuda()

    return train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
        headTailSelector_list, unique_entities_train_list, rel_embed_list, ent_freq_mat, nentity


