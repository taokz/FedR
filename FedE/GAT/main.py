import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import copy
import random
import argparse
import os
import sys
import logging
import time
import pickle
import json

from fedGAT import fedGAT
from GAT import GAT


def parse_args():
    args = argparse.ArgumentParser()

    # logging arguments
    args.add_argument('--data_path', default='Fed_data/DDB14-Fed3.pkl', type=str)
    args.add_argument('--name', default='ddb14_fed3_fed', type=str)
    args.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    args.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    args.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)
    args.add_argument('--run_mode', default='FedGAT', choices=['FedGAT', 'Single', 'Entire'])

    # federated arguments
    args.add_argument('--num_client', default=3, type=int)
    args.add_argument('--max_round', default=1000, type=int) #10000
    args.add_argument('--fraction', default=1, type=float)
    args.add_argument('--log_per_round', default=1, type=int)
    args.add_argument('--check_per_round', default=5, type=int)

    args.add_argument('--early_stop_patience', default=5, type=int)

    # network arguments
    # args.add_argument("-data", "--data",
    #                   default="./data/WN18RR", help="data directory") # ./data/WN18RR
    args.add_argument('--gpu', default='0', type=str)
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=500, help="Number of epochs") # 3600
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=150, help="Number of epochs") # 200
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)") # 50
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="checkpoint/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=512, help="Batch size for GAT") # 86835
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[50, 100], help="Entity output embedding dimensions") # 100, 200 -> 50, 100
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[1, 1], help="Multihead attention SpGAT") # 2, 2
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=50,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")

    args.add_argument('--seed', default=12345, type=int)

    args = args.parse_args()
    return args

def init_dir(args):
    # state
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


def init_logger(args):
    log_file = os.path.join(args.log_dir, args.name + '.log')

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode='a+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

if __name__ == '__main__':
    args = parse_args()
    args_str = json.dumps(vars(args))

    args.gpu = torch.device('cuda:' + args.gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    init_dir(args)
    writer = SummaryWriter(os.path.join(args.tb_log_dir, args.name))
    args.writer = writer
    init_logger(args)
    logging.info(args_str)

    if args.run_mode == 'FedGAT':
        all_data = pickle.load(open(args.data_path, 'rb'))
        learner = fedGAT(args, all_data)
        learner.train()
    elif args.run_mode == 'Single':
        all_data = pickle.load(open(args.data_path, 'rb'))
        learner = GAT(args, all_data)
        learner.train()
    elif args.run_mode == 'Entire':
        all_data = pickle.load(open(args.data_path, 'rb'))
        learner = GAT(args, all_data)
        learner.train() 
