from preprocess import get_all_clients
from create_batch import Corpus
from models import SpKBGATModified, SpKBGATConvOnly
#from utils import save_model

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import logging
import os
import argparse
import json
import pickle
import numpy as np
import copy
import random
from collections import defaultdict as ddict

class Server(object):
    def __init__(self, args, nentity):
        self.args = args
        self.entity_embeddings = torch.Tensor(np.zeros((nentity, args.embedding_size))).cuda()
        self.nentity = nentity

    def send_emb(self):
        return copy.deepcopy(self.entity_embeddings)

    def aggregation(self, clients, ent_update_weights):
        agg_ent_mask = ent_update_weights
        agg_ent_mask[ent_update_weights != 0] = 1

        ent_w_sum = torch.sum(agg_ent_mask, dim=0)
        ent_w = agg_ent_mask / ent_w_sum
        ent_w[torch.isnan(ent_w)] = 0

        update_ent_embed = torch.Tensor(np.zeros((self.nentity, self.args.embedding_size))).cuda()
        for i, client in enumerate(clients):
            local_ent_embed = client.entity_embeddings.clone().detach()
            # print(local_rel_embed.shape, rel_w[i].shape)
            update_ent_embed += local_ent_embed * ent_w[i].reshape(-1, 1)
        self.entity_embeddings = update_ent_embed

class Client(object):
    def __init__(self, args, client_id, train_dataloader,
                 valid_dataloader, test_dataloader, headTailSelector, unique_entities_train, rel_embed, ent_embed):
        self.args = args
        # self.data = data
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.client_id = client_id
        
        nentity = ent_embed.shape[0]
        # self.entity_embeddings = torch.FloatTensor(ent_embed)
        self.relation_embeddings = torch.Tensor(rel_embed)
        self.entity_embeddings = ent_embed
        self.Corpus_ = Corpus(args, train_dataloader, valid_dataloader, test_dataloader, headTailSelector, 
        					 args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, nentity, args.get_2hop)
        if args.use_2hop:
        	self.node_neighbors_2hop = self.Corpus_.node_neighbors_2hop
        # self.current_batch_2hop_indices = self.Corpus_.get_batch_nhop_neighbors_all(self.args,
        #                                                                 self.Corpus_.unique_entities_train, self.node_neighbors_2hop)

        self.model_gat = SpKBGATModified(self.entity_embeddings, self.relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, args.gpu)
        self.model_conv = SpKBGATConvOnly(self.entity_embeddings, self.relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                     args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                     args.nheads_GAT, args.out_channels)

        self.model_conv.cuda()
        self.model_gat.cuda()

    def batch_gat_loss(self, gat_loss_func, train_indices, entity_embed, relation_embed):
        len_pos_triples = int(
            train_indices.shape[0] / (int(self.args.valid_invalid_ratio_gat) + 1))

        pos_triples = train_indices[:len_pos_triples]
        neg_triples = train_indices[len_pos_triples:]

        pos_triples = pos_triples.repeat(int(self.args.valid_invalid_ratio_gat), 1)

        if (len(pos_triples) > len(neg_triples)):
        	pos_triples = pos_triples[:len(neg_triples)]
        elif (len(pos_triples) < len(neg_triples)):
        	neg_triples = neg_triples[:len(pos_triples)]

        source_embeds = entity_embed[pos_triples[:, 0]]
        relation_embeds = relation_embed[pos_triples[:, 1]]
        tail_embeds = entity_embed[pos_triples[:, 2]]

        x = source_embeds + relation_embeds - tail_embeds
        pos_norm = torch.norm(x, p=1, dim=1)

        source_embeds = entity_embed[neg_triples[:, 0]]
        relation_embeds = relation_embed[neg_triples[:, 1]]
        tail_embeds = entity_embed[neg_triples[:, 2]]

        x = source_embeds + relation_embeds - tail_embeds
        neg_norm = torch.norm(x, p=1, dim=1)

        y = -torch.ones(int(self.args.valid_invalid_ratio_gat) * len_pos_triples).cuda()
        # y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).to(args.gpu)

        loss = gat_loss_func(pos_norm, neg_norm, y)
        return loss

    def train_gat(self):

        self.model_gat.entity_embeddings.data = copy.deepcopy(self.entity_embeddings)

        optimizer = torch.optim.Adam(
            self.model_gat.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay_gat)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=500, gamma=0.5, last_epoch=-1)

        gat_loss_func = nn.MarginRankingLoss(margin=self.args.margin)

        current_batch_2hop_indices = torch.tensor([])
        if(self.args.use_2hop):
            current_batch_2hop_indices = self.Corpus_.get_batch_nhop_neighbors_all(self.args,
                                                                              self.Corpus_.unique_entities_train, self.node_neighbors_2hop)

        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()
        # current_batch_2hop_indices = Variable(
        #     torch.LongTensor(current_batch_2hop_indices)).to(args.gpu)

        epoch_losses = []   # losses of all epochs

        for epoch in range(self.args.epochs_gat):
            random.shuffle(self.Corpus_.train_triples)
            self.Corpus_.train_indices = np.array(
                list(self.Corpus_.train_triples)).astype(np.int32)

            self.model_gat.train()  # getting in training mode
            # start_time = time.time()
            epoch_loss = []

            if len(self.Corpus_.train_indices) % self.args.batch_size_gat == 0:
                num_iters_per_epoch = len(
                    self.Corpus_.train_indices) // self.args.batch_size_gat
            else:
                num_iters_per_epoch = (
                    len(self.Corpus_.train_indices) // self.args.batch_size_gat) + 1

            for iters in range(num_iters_per_epoch):
                # start_time_iter = time.time()
                train_indices, train_values = self.Corpus_.get_iteration_batch(iters)

                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()
                # train_indices = Variable(
                #     torch.LongTensor(train_indices)).to(args.gpu)
                # train_values = Variable(torch.FloatTensor(train_values)).to(args.gpu)

                # forward pass
                entity_embed, relation_embed = self.model_gat(
                    self.Corpus_, self.Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

                optimizer.zero_grad()

                loss = self.batch_gat_loss(
                    gat_loss_func, train_indices, entity_embed, relation_embed)

                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.data.item())

                # end_time_iter = time.time()

            scheduler.step()

            epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        self.entity_embeddings = self.model_gat.final_entity_embeddings.data.clone().detach()
        self.relation_embeddings = self.model_gat.final_relation_embeddings.data.clone().detach()

        return (sum(epoch_losses)/len(epoch_losses))


    def train_conv(self):

        # self.model_conv.final_entity_embeddings = self.model_gat.final_entity_embeddings
        # self.model_conv.final_relation_embeddings = self.model_gat.final_relation_embeddings

        self.Corpus_.batch_size = self.args.batch_size_conv
        self.Corpus_.invalid_valid_ratio = int(self.args.valid_invalid_ratio_conv)

        optimizer = torch.optim.Adam(
            self.model_conv.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay_conv)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=25, gamma=0.5, last_epoch=-1)

        margin_loss = torch.nn.SoftMarginLoss()

        epoch_losses = []   # losses of all epochs
        # print("Number of epochs {}".format(args.epochs_conv))

        for epoch in range(self.args.epochs_conv):
            # print("\nepoch-> ", epoch)
            random.shuffle(self.Corpus_.train_triples)
            self.Corpus_.train_indices = np.array(
                list(self.Corpus_.train_triples)).astype(np.int32)

            self.model_conv.train()  # getting in training mode
            # start_time = time.time()
            epoch_loss = []

            if len(self.Corpus_.train_indices) % self.args.batch_size_conv == 0:
                num_iters_per_epoch = len(
                    self.Corpus_.train_indices) // self.args.batch_size_conv
            else:
                num_iters_per_epoch = (
                    len(self.Corpus_.train_indices) // self.args.batch_size_conv) + 1

            for iters in range(num_iters_per_epoch):
                # start_time_iter = time.time()
                train_indices, train_values = self.Corpus_.get_iteration_batch(iters)

                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

                preds = self.model_conv(
                    self.Corpus_, self.Corpus_.train_adj_matrix, train_indices)

                optimizer.zero_grad()

                loss = margin_loss(preds.view(-1), train_values.view(-1))

                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.data.item())

            scheduler.step()
            epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

            # save_model(model_conv, args.name, num_round,
            #            args.output_folder, self.client_id)

        # self.entity_embeddings = self.model_conv.final_entity_embeddings.data.clone().detach()
        # self.relation_embeddings = self.model_conv.final_relation_embeddings.data.clone().detach()

        # return (sum(epoch_losses)/len(epoch_losses))

    def client_update(self):
        loss = self.train_gat()
        return loss

    def client_eval(self, istest):

        self.model_conv.final_relation_embeddings.data = self.relation_embeddings
        self.model_conv.final_entity_embeddings.data = self.entity_embeddings

        self.train_conv()

        self.model_conv.eval()
        with torch.no_grad():
            result = self.Corpus_.get_validation_pred(self.args, self.model_conv, self.Corpus_.unique_entities_train, istest)

        return result


class fedGAT(object):
    def __init__(self, args, all_data):
        train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
            headTailSelector_list, unique_entities_train_list, rel_embed_list, \
            self.ent_freq_mat, nentity = get_all_clients(all_data, args)

        self.nentity = nentity
        self.args = args

        # client
        self.num_clients = len(train_dataloader_list)
        self.server = Server(args, nentity)
        # self.clients = [
        #     Client(args, i, all_data[i], train_dataloader_list[i], valid_dataloader_list[i],
        #            test_dataloader_list[i], headTailSelector_list[i], unique_entities_train_list[i], rel_embed_list[i], self.server.entity_embeddings) for i in range(self.num_clients)
        # ]
        self.clients = [
            Client(args, i, train_dataloader_list[i], valid_dataloader_list[i],
                   test_dataloader_list[i], headTailSelector_list[i], unique_entities_train_list[i], rel_embed_list[i], self.server.entity_embeddings) for i in range(self.num_clients)
        ]

        self.total_test_data_size = sum([len(client.test_dataloader[0]) for client in self.clients])
        self.test_eval_weights = [len(client.test_dataloader[0]) / self.total_test_data_size for client in self.clients]

        self.total_valid_data_size = sum([len(client.valid_dataloader[0]) for client in self.clients])
        self.valid_eval_weights = [len(client.valid_dataloader[0]) / self.total_valid_data_size for client in self.clients]

    def write_training_loss(self, loss, e):
        self.args.writer.add_scalar("training/loss", loss, e)

    def write_evaluation_result(self, results, e):
        self.args.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.args.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.args.writer.add_scalar("evaluation/hits3", results['hits@3'], e)
        self.args.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def save_checkpoint(self, e):
        state = {'ent_embed': self.server.entity_embeddings,
                 'rel_embed': [client.relation_embeddings for client in self.clients]}
        # delete previous checkpoint
        for filename in os.listdir(self.args.state_dir):
            if self.args.name in filename.split('.') and os.path.isfile(os.path.join(self.args.state_dir, filename)):
                os.remove(os.path.join(self.args.state_dir, filename))
        # save current checkpoint
        torch.save(state, os.path.join(self.args.state_dir,
                                       self.args.name + '.' + str(e) + '.ckpt'))
        # delete previous models checkpoints
        for filename in os.listdir(self.args.output_folder):
            if self.args.name in filename.split('.') and os.path.isfile(os.path.join(self.args.output_folder, filename)):
                os.remove(os.path.join(self.args.output_folder, filename))
        # save current models checkpoints
        for client in self.clients:
        	torch.save(client.model_conv.state_dict(), os.path.join(self.args.output_folder, self.args.name + '.' + str(e) + '_client.' + str(client.client_id) + '.pth'))

    def save_model(self, best_epoch):
        os.rename(os.path.join(self.args.state_dir, self.args.name + '.' + str(best_epoch) + '.ckpt'),
                  os.path.join(self.args.state_dir, self.args.name + '.best'))
        for client in self.clients:
            os.rename(os.path.join(self.args.output_folder, self.args.name + '.' + str(best_epoch) + '_client.' + str(client.client_id) + '.pth'),
                      os.path.join(self.args.output_folder, self.args.name + '_client.' + str(client.client_id) + '.best'))

    def send_emb(self):
        for k, client in enumerate(self.clients):
            client.entity_embeddings = self.server.send_emb()

    def train(self):
        best_epoch = 0
        best_mrr = 0
        bad_count = 0

        mrr_plot_result = []
        loss_plot_result = []

        for num_round in range(self.args.max_round):
            n_sample = max(round(self.args.fraction * self.num_clients), 1)
            sample_set = np.random.choice(self.num_clients, n_sample, replace=False)

            self.send_emb()
            
            round_loss = 0
            for k in iter(sample_set):
                client_loss = self.clients[k].client_update()
                round_loss += client_loss
            round_loss /= n_sample
            self.server.aggregation(self.clients, self.ent_freq_mat)

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
    	# load embeddings
        state = torch.load(os.path.join(self.args.state_dir, self.args.name + '.best'), map_location=self.args.gpu)
        self.server.entity_embeddings = state['ent_embed']
        for idx, client in enumerate(self.clients):
            client.relation_embeddings = state['rel_embed'][idx]
        # load conv models
        for client in self.clients:
        	PATH = os.path.join(self.args.output_folder, self.args.name + '_client.' + str(client.client_id) + '.best')
        	client.model_conv.load_state_dict(torch.load(PATH))


    def evaluate(self, istest=False):
        self.send_emb()
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