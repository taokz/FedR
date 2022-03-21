import torch
#from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import DataLoader
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

from fedNoGE import fedNoGE
from NoGE import NoGE


def parse_args():
    args = argparse.ArgumentParser()

    # logging arguments
    args.add_argument('--data_path', default='Fed_data/DDB14-Fed3.pkl', type=str)
    args.add_argument('--name', default='ddb14_fed3_fed', type=str)
    args.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    args.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    args.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)
    args.add_argument('--run_mode', default='FedNoGE', choices=['FedNoGE', 'Single', 'Entire'])

    # federated arguments
    args.add_argument('--num_client', default=3, type=int)
    args.add_argument('--max_round', default=1000, type=int) #10000
    args.add_argument('--fraction', default=1, type=float)
    args.add_argument('--log_per_round', default=1, type=int)
    args.add_argument('--check_per_round', default=5, type=int)

    args.add_argument('--early_stop_patience', default=5, type=int)

    args.add_argument('--local_epoch', default=3, type=int)

    # network arguments
    # args.add_argument("-data", "--data",
    #                   default="./data/WN18RR", help="data directory") # ./data/WN18RR
    args.add_argument('--gpu', default='0', type=str)
    args.add_argument('--seed', default=12345, type=int)

    # arguments for NoGE
    args.add_argument('--learning_rate', default=0.01, type=float)
    args.add_argument('--hid_dim', default=128, type=int)
    args.add_argument('--emb_dim', default=128, type=int)
    args.add_argument('--batch_size', default=1024, type=int)
    args.add_argument('--label_smoothing', default=0.1, type=float)
    args.add_argument('--num_iterations', default=3000, type=int)
    args.add_argument('--num_layers', default=1, type=int)
    args.add_argument('--eval_step', default=1, type=int)
    args.add_argument('--eval_after', default=1000, type=int)

    # arguments for SecureAgg
    args.add_argument('--isSecure', default=0, type=int)

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

    if args.run_mode == 'FedNoGE':
        all_data = pickle.load(open(args.data_path, 'rb'))
        learner = fedNoGE(args, all_data)
        learner.train()
    elif args.run_mode == 'Single':
        all_data = pickle.load(open(args.data_path, 'rb'))
        learner = NoGE(args, all_data)
        learner.train()
    elif args.run_mode == 'Entire':
        all_data = pickle.load(open(args.data_path, 'rb'))
        learner = NoGE(args, all_data)
        learner.train() 