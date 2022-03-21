import logging
import os
import argparse
import json
import pickle
import numpy as np
import copy
import random
from collections import defaultdict as ddict
from functools import reduce
import torch
import time
#from collections import defaultdict
import scipy.sparse as sp

from utils_NoGE import *
from models import NoGE_GCN_QuatE
from dataloader import get_all_clients

class Client(object):
    def __init__(self, args, client_id, data, train_dataloader,
                valid_dataloader, test_dataloader, rel_embed, train_adj, nentity):
        self.args = args
        self.data = data # all triples
        self.rel_embed = rel_embed
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.client_id = client_id
        self.train_adj = train_adj
        self.nentity = nentity

        self.kge_model = NoGE_GCN_QuatE(args.emb_dim, args.hid_dim, train_adj, nentity, rel_embed.shape[0], self.args.num_layers).to(args.gpu)
        # self.ent_embed = None
        # self.rel_embed = rel_embed

    def __len__(self):
        return len(self.train_dataloader)

    def client_update(self):
        train_data_idxs = self.get_data_idxs(self.train_dataloader)
        adj = self.train_adj

        opt = torch.optim.Adam(self.kge_model.parameters(), lr=self.args.learning_rate)

        lst_indexes = torch.LongTensor([i for i in range(self.nentity + self.rel_embed.shape[0])]).to(self.args.gpu)
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        max_valid = 0.0
        for it in range(1, self.args.num_iterations + 1):
            self.kge_model.train()
            train_time = []
            losses = []
            start = time.time()
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.args.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0]).to(self.args.gpu)
                r_idx = torch.tensor(data_batch[:, 1]).to(self.args.gpu)

                predictions = self.kge_model.forward(e1_idx, r_idx, lst_indexes)
                if self.args.label_smoothing:
                    targets = ((1.0 - self.args.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = self.kge_model.loss(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.kge_model.parameters(), 0.5)  # prevent the exploding gradient problem
                opt.step()
                losses.append(loss.item())
            end = time.time()
            train_time.append(end-start)
            if it % 100 == 0:
                logging.info("Epoch: {}, --> Loss: {:.4f}, --> Time: {:.8f}".format(it, np.mean(losses), np.mean(train_time)))
            
            # evaluation
            if it > self.args.eval_after and it % self.args.eval_step == 0:
                tmp_mrr = self.client_eval(istest=False)['mrr']
                if max_valid < tmp_mrr:
                    max_valid = tmp_mrr
                    best_epoch = it
                else:
                    break

        
        #return np.mean(train_time)
        #return np.mean(losses)

    def get_data_idxs(self, data):
        data_idxs = [(data[i][0], data[i][1], data[i][2]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = ddict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.args.batch_size]
        targets = np.zeros((len(batch), self.nentity))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        return np.array(batch), targets.to(self.args.gpu)

    def client_eval(self, istest=False):
        if istest:
            data = self.test_dataloader
        else:
            data = self.valid_dataloader

        lst_indexes = torch.LongTensor([i for i in range(self.nentity + self.rel_embed.shape[0])]).to(self.args.gpu)

        results = ddict(float)

        self.kge_model.eval()
        with torch.no_grad():
            hits = []
            ranks = []
            for i in range(10):
                hits.append([])

            test_data_idxs = self.get_data_idxs(data)
            er_vocab = self.get_er_vocab(self.get_data_idxs(self.data))

            for i in range(0, len(test_data_idxs), self.args.batch_size):
                data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
                e1_idx = torch.tensor(data_batch[:, 0]).to(self.args.gpu)
                r_idx = torch.tensor(data_batch[:, 1]).to(self.args.gpu)
                e2_idx = torch.tensor(data_batch[:, 2]).to(self.args.gpu)

                predictions = self.kge_model.forward(e1_idx, r_idx, lst_indexes).detach()

                for j in range(data_batch.shape[0]):
                    filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                    target_value = predictions[j, e2_idx[j]].item()
                    predictions[j, filt] = 0.0
                    predictions[j, e2_idx[j]] = target_value

                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

                sort_idxs = sort_idxs.cpu().numpy()
                for j in range(data_batch.shape[0]):
                    rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                    ranks.append(rank + 1)
                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)

        results['mrr'] = np.mean(1. / np.array(ranks))
        for k in [1, 3, 10]:
            results['hits@{}'.format(k)] = np.mean(hits[k-1]) #* 100

        return results

class NoGE(object):
    def __init__(self, args, all_data):
        self.args = args

        train_dataloader_list, valid_dataloader_list, test_dataloader_list, all_dataloader_list,\
            self.ent_freq_mat, rel_embed_list, train_adj_list, nentity = get_all_clients(all_data, args)

        self.args.nentity = nentity

        # client
        self.num_clients = len(train_dataloader_list)
        self.clients = [
            Client(args, i, all_dataloader_list[i], train_dataloader_list[i], valid_dataloader_list[i],
                   test_dataloader_list[i], rel_embed_list[i], train_adj_list[i], nentity) for i in range(self.num_clients)
        ]

    def train(self):

        result_mrr = []
        result_h1 = []
        result_h3 = []
        result_h10 = []

        for k in range(self.args.num_client):
            self.clients[k].client_update()
            client_res = self.clients[k].client_eval(istest=True)

            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
                client_res['mrr'], client_res['hits@1'],
                client_res['hits@3'], client_res['hits@10']))

            result_mrr.append(client_res['mrr'])
            result_h1.append(client_res['hits@1'])
            result_h3.append(client_res['hits@3'])
            result_h10.append(client_res['hits@10'])

        logging.info('avg mrr: {:.4f}, avg hits@1: {:.4f}, avg hits@3: {:.4f}, avg hits@10: {:.4f}'.format(
            sum(result_mrr)/len(result_mrr), sum(result_h1)/len(result_h1), 
            sum(result_h3)/len(result_h3), sum(result_h10)/len(result_h10)))