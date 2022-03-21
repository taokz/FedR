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


class Server(object):
    def __init__(self, args, nrelation):
        self.args = args
        self.relation_embeddings = torch.Tensor(np.zeros((nrelation, args.emb_dim))).to(args.gpu)
        self.nrelation = nrelation

    def send_emb(self, init):
        if self.args.isSecure == 1:
            if init == 0:
                server_rel_emb = copy.deepcopy(self.relation_embeddings)
            elif init == 1:
                server_rel_emb = copy.deepcopy(self.relation_embeddings)
                server_rel_emb = server_rel_emb / (10**self.args.quantization_bit)
        else:
            server_rel_emb = copy.deepcopy(self.relation_embeddings)
        return server_rel_emb

    def aggregation(self, clients, rel_update_weights):
        agg_rel_mask = rel_update_weights
        agg_rel_mask[rel_update_weights != 0] = 1

        rel_w_sum = torch.sum(agg_rel_mask, dim=0)
        rel_w = agg_rel_mask / rel_w_sum
        rel_w[torch.isnan(rel_w)] = 0

        update_rel_embed = torch.Tensor(np.zeros((self.nrelation, self.args.emb_dim))).to(self.args.gpu)

        # PSI
        if self.args.isSecure == 2:
            mask_index = [np.nonzero(rel_w[i]).cpu().detach().numpy() for i in range(self.args.num_client)]
            mask_psi_ind = reduce(np.intersect1d, mask_index)

        for i, client in enumerate(clients):
            local_rel_embed = client.rel_embed.clone().detach()
            if self.args.isSecure == 1:
                local_rel_embed = (local_rel_embed * (10**self.args.quantization_bit)).long().float()
            update_rel_embed += local_rel_embed * rel_w[i].reshape(-1, 1)
        
        if self.args.isSecure == 2:
            self.relation_embeddings[mask_psi_ind] = update_rel_embed[mask_psi_ind]
        else:
            self.relation_embeddings = update_rel_embed

class Client(object):
    def __init__(self, args, client_id, data, train_dataloader,
                valid_dataloader, test_dataloader, ent_embed, train_adj, nrelation):
        self.args = args
        self.data = data # all triples
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.ent_embed = ent_embed
        self.client_id = client_id
        self.train_adj = train_adj
        self.nrelation = nrelation

        self.kge_model = NoGE_GCN_QuatE(args.emb_dim, args.hid_dim, train_adj, ent_embed.shape[0], nrelation, self.args.num_layers).to(args.gpu)
        self.rel_embed = None

    def __len__(self):
        return len(self.train_dataloader)

    def client_update(self):
        train_data_idxs = self.get_data_idxs(self.train_dataloader)
        adj = self.train_adj

        opt = torch.optim.Adam(self.kge_model.parameters(), lr=self.args.learning_rate)

        lst_indexes = torch.LongTensor([i for i in range(self.ent_embed.shape[0] + self.nrelation)]).to(self.args.gpu)
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        losses = []
        for it in range(1, self.args.local_epoch + 1):
            self.kge_model.train()
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

        # self.ent_embed = copy.deepcopy(self.kge_model.embeddings.weight.data[:self.ent_embed.shape[0]])
        # self.rel_embed = copy.deepcopy(self.kge_model.embeddings.weight.data[self.ent_embed.shape[0]:])
        self.rel_embed = copy.deepcopy(self.kge_model.embeddings.weight.data[:self.nrelation])
        self.ent_embed = copy.deepcopy(self.kge_model.embeddings.weight.data[self.nrelation:])

        return np.mean(losses)

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
        targets = np.zeros((len(batch), self.ent_embed.shape[0]))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        return np.array(batch), targets.to(self.args.gpu)

    def client_eval(self, istest=False):
        if istest:
            data = self.test_dataloader
        else:
            data = self.valid_dataloader

        lst_indexes = torch.LongTensor([i for i in range(self.ent_embed.shape[0] + self.nrelation)]).to(self.args.gpu)

        results = ddict(float)

        self.kge_model.eval()
        # self.kge_model.embeddings.weight.data[:self.ent_embed.shape[0]] = self.ent_embed
        # self.kge_model.embeddings.weight.data[self.ent_embed.shape[0]:] = self.rel_embed
        self.kge_model.embeddings.weight.data[self.nrelation:] = self.ent_embed
        self.kge_model.embeddings.weight.data[:self.nrelation] = self.rel_embed
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
            results['hits@{}'.format(k)] = np.mean(hits[k-1]) #* 100 # I don't know why

        return results


class fedNoGE(object):
    def __init__(self, args, all_data):
        self.args = args

        train_dataloader_list, valid_dataloader_list, test_dataloader_list, all_dataloader_list,\
            self.rel_freq_mat, ent_embed_list, train_adj_list, nrelation = get_all_clients(all_data, args)

        self.args.nrelation = nrelation

        # client
        self.num_clients = len(train_dataloader_list)
        self.clients = [
            Client(args, i, all_dataloader_list[i], train_dataloader_list[i], valid_dataloader_list[i],
                   test_dataloader_list[i], ent_embed_list[i], train_adj_list[i], nrelation) for i in range(self.num_clients)
        ]

        self.server = Server(args, nrelation)

        self.total_test_data_size = sum([len(client.test_dataloader) for client in self.clients])
        self.test_eval_weights = [len(client.test_dataloader) / self.total_test_data_size for client in self.clients]

        self.total_valid_data_size = sum([len(client.valid_dataloader) for client in self.clients])
        self.valid_eval_weights = [len(client.valid_dataloader) / self.total_valid_data_size for client in self.clients]

    def write_training_loss(self, loss, e):
        self.args.writer.add_scalar("training/loss", loss, e)

    def write_evaluation_result(self, results, e):
        self.args.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.args.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.args.writer.add_scalar("evaluation/hits3", results['hits@3'], e)
        self.args.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def save_checkpoint(self, e):
        state = {'rel_embed': self.server.relation_embeddings,
                 'ent_embed': [client.ent_embed for client in self.clients]}
        # delete previous checkpoint
        for filename in os.listdir(self.args.state_dir):
            if self.args.name in filename.split('.') and os.path.isfile(os.path.join(self.args.state_dir, filename)):
                os.remove(os.path.join(self.args.state_dir, filename))
        # save current checkpoint
        torch.save(state, os.path.join(self.args.state_dir,
                                       self.args.name + '.' + str(e) + '.ckpt'))

    def save_model(self, best_epoch):
        os.rename(os.path.join(self.args.state_dir, self.args.name + '.' + str(best_epoch) + '.ckpt'),
                  os.path.join(self.args.state_dir, self.args.name + '.best'))

    def send_emb(self, init):
        for k, client in enumerate(self.clients):
            client.rel_embed = self.server.send_emb(init)

    def train(self):
        best_epoch = 0
        best_mrr = 0
        bad_count = 0

        mrr_plot_result = []
        loss_plot_result = []

        for num_round in range(self.args.max_round):
            n_sample = max(round(self.args.fraction * self.num_clients), 1)
            sample_set = np.random.choice(self.num_clients, n_sample, replace=False)

            if num_round == 0:
                self.send_emb(init = 0)
            else:
                self.send_emb(init = 1)
            
            round_loss = 0
            for k in iter(sample_set):
                client_loss = self.clients[k].client_update()
                round_loss += client_loss
            round_loss /= n_sample
            self.server.aggregation(self.clients, self.rel_freq_mat)

            logging.info('round: {} | loss: {:.4f}'.format(num_round, np.mean(round_loss)))
            self.write_training_loss(np.mean(round_loss), num_round)

            loss_plot_result.append(np.mean(round_loss))

            if num_round % self.args.check_per_round == 0 and num_round != 0:
                eval_res = self.evaluate()
                self.write_evaluation_result(eval_res, num_round)

                if eval_res['mrr'] > best_mrr:
                    best_mrr = eval_res['mrr']
                    best_epoch = num_round
                    logging.info('best model | mrr {:.4f}'.format(best_mrr))
                    self.save_checkpoint(num_round)
                    bad_count = 0
                else:
                    bad_count += 1
                    logging.info('best model is at round {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_epoch, best_mrr, bad_count))

                mrr_plot_result.append(eval_res['mrr'])

            if bad_count >= self.args.early_stop_patience:
                logging.info('early stop at round {}'.format(num_round))

                loss_file_name = 'loss/' + self.args.name + '_loss.pkl'
                with open(loss_file_name, 'wb') as fp:
                    pickle.dump(loss_plot_result, fp)

                mrr_file_name = 'loss/' + self.args.name + '_mrr.pkl'
                with open(mrr_file_name, 'wb') as fp:
                    pickle.dump(mrr_plot_result, fp)
                break

        logging.info('finish training')
        logging.info('save best model')
        self.save_model(best_epoch)
        self.before_test_load()
        self.evaluate(istest=True)

    def before_test_load(self):
        state = torch.load(os.path.join(self.args.state_dir, self.args.name + '.best'), map_location=self.args.gpu)
        self.server.relation_embeddings = state['rel_embed']
        for idx, client in enumerate(self.clients):
            client.ent_embed = state['ent_embed'][idx]

    def evaluate(self, istest=False):
        self.send_emb(init=1)
        result = ddict(int)
        if istest:
            weights = self.test_eval_weights
        else:
            weights = self.valid_eval_weights
        for idx, client in enumerate(self.clients):
            client_res = client.client_eval(istest)

            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
                client_res['mrr'], client_res['hits@1'],
                client_res['hits@3'], client_res['hits@10']))

            for k, v in client_res.items():
                result[k] += v * weights[idx]

        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
                     result['mrr'], result['hits@1'],
                     result['hits@3'], result['hits@10']))

        return result