from preprocess import get_all_clients
from create_batch import Corpus
from models import SpKBGATModified, SpKBGATConvOnly

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
import time

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

            # epoch_losses.append(sum(epoch_loss) / len(epoch_loss))


    def train_conv(self):

        self.model_conv.final_entity_embeddings = self.model_gat.final_entity_embeddings
        self.model_conv.final_relation_embeddings = self.model_gat.final_relation_embeddings

        self.Corpus_.batch_size = self.args.batch_size_conv
        self.Corpus_.invalid_valid_ratio = int(self.args.valid_invalid_ratio_conv)

        optimizer = torch.optim.Adam(
            self.model_conv.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay_conv)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=25, gamma=0.5, last_epoch=-1)

        margin_loss = torch.nn.SoftMarginLoss()

        epoch_losses = []   # losses of all epochs
        print("Number of epochs {}".format(self.args.epochs_conv))

        for epoch in range(self.args.epochs_conv):
            print("\nepoch-> ", epoch)
            random.shuffle(self.Corpus_.train_triples)
            self.Corpus_.train_indices = np.array(
                list(self.Corpus_.train_triples)).astype(np.int32)

            self.model_conv.train()  # getting in training mode
            
            start_time = time.time()
            epoch_loss = []

            if len(self.Corpus_.train_indices) % self.args.batch_size_conv == 0:
                num_iters_per_epoch = len(
                    self.Corpus_.train_indices) // self.args.batch_size_conv
            else:
                num_iters_per_epoch = (
                    len(self.Corpus_.train_indices) // self.args.batch_size_conv) + 1

            for iters in range(num_iters_per_epoch):
                start_time_iter = time.time()
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

                end_time_iter = time.time()

                print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                    iters, end_time_iter - start_time_iter, loss.data.item()))

            scheduler.step()
            print("Epoch {} , average loss {} , epoch_time {}".format(
                epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
            # epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

            # save_model(model_conv, args.name, num_round,
            #            args.output_folder, self.client_id)

        # self.entity_embeddings = self.model_conv.final_entity_embeddings.data.clone().detach()
        # self.relation_embeddings = self.model_conv.final_relation_embeddings.data.clone().detach()
        # return (sum(epoch_losses)/len(epoch_losses))

    def client_update(self):
        self.train_gat()
        # loss = self.train_conv()
        self.train_conv()
        # return loss

    def client_eval(self, istest):

        # if istest:
        #     self.model_conv.final_relation_embeddings.data = self.relation_embeddings
        #     self.model_conv.final_entity_embeddings.data = self.entity_embeddings

        # self.model_conv.final_entity_embeddings.data = self.entity_embeddings

        self.model_conv.eval()
        with torch.no_grad():
            result = self.Corpus_.get_validation_pred(self.args, self.model_conv, self.Corpus_.unique_entities_train, istest)

        return result


class GAT(object):
    def __init__(self, args, all_data):
        train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
            headTailSelector_list, unique_entities_train_list, rel_embed_list, \
            self.ent_freq_mat, nentity = get_all_clients(all_data, args)

        self.nentity = nentity
        self.args = args

        # client
        self.num_clients = len(train_dataloader_list)
        self.entity_embeddings = torch.Tensor(np.zeros((nentity, args.embedding_size))).cuda()
        self.clients = [
            Client(args, i, train_dataloader_list[i], valid_dataloader_list[i],
                   test_dataloader_list[i], headTailSelector_list[i], unique_entities_train_list[i], 
                   rel_embed_list[i], self.entity_embeddings) for i in range(self.num_clients)
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
            sum(result_h3)/len(result_h3), sum(result_h10)/len(result_h10))