import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict as ddict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size

        self.hr2t = ddict(set)
        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
        for h, r in self.hr2t:
            self.hr2t[(h, r)] = np.array(list(self.hr2t[(h, r)]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.hr2t[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, idx

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        sample_idx = torch.tensor([_[2] for _ in data])
        return positive_sample, negative_sample, sample_idx


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, rel_mask = None):
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples
        self.nentity = nentity

        self.rel_mask = rel_mask

        self.hr2t_all = ddict(set)
        for h, r, t in all_true_triples:
            self.hr2t_all[(h, r)].add(t)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return triple, trp_label

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        label = self.hr2t_all[(head, relation)]
        trp_label = self.get_label(label)
        triple = torch.LongTensor((head, relation, tail))

        return triple, trp_label

    def get_label(self, label):
        y = np.zeros([self.nentity], dtype=np.float32)

        for e2 in label:
            y[e2] = 1.0

        return torch.FloatTensor(y)


def get_task_dataset(data, args):
    nentity = len(np.unique(data['train']['edge_index'].reshape(-1)))
    nrelation = len(np.unique(data['train']['edge_type']))

    train_triples = np.stack((data['train']['edge_index'][0],
                              data['train']['edge_type'],
                              data['train']['edge_index'][1])).T

    valid_triples = np.stack((data['valid']['edge_index'][0],
                              data['valid']['edge_type'],
                              data['valid']['edge_index'][1])).T

    test_triples = np.stack((data['test']['edge_index'][0],
                             data['test']['edge_type'],
                             data['test']['edge_index'][1])).T

    all_triples = np.concatenate([train_triples, valid_triples, test_triples])
    train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
    valid_dataset = TestDataset(valid_triples, all_triples, nentity)
    test_dataset = TestDataset(test_triples, all_triples, nentity)

    return train_dataset, valid_dataset, test_dataset, nrelation, nentity


def get_all_clients(all_data, args):
    all_rel = np.array([], dtype=int)
    for data in all_data:
        all_rel = np.union1d(all_rel, data['train']['edge_type_ori']).reshape(-1)
    nrelation = len(all_rel) # all relations of training set in all clients

    train_dataloader_list = []
    test_dataloader_list = []
    valid_dataloader_list = []

    ent_embed_list = []

    rel_freq_list = []

    for data in tqdm(all_data): # in a client
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

        all_triples = np.concatenate([train_triples, valid_triples, test_triples]) # in a client
        train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
        valid_dataset = TestDataset(valid_triples, all_triples, nentity, client_mask_rel)
        test_dataset = TestDataset(test_triples, all_triples, nentity, client_mask_rel)

        # dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_list.append(train_dataloader)

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        valid_dataloader_list.append(valid_dataloader)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_list.append(test_dataloader)

        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])

        '''use n of entity in train or all (train, valid, test)?'''
        if args.model in ['RotatE', 'ComplEx']:
            ent_embed = torch.zeros(nentity, args.hidden_dim*2).to(args.gpu).requires_grad_()
        else:
            ent_embed = torch.zeros(nentity, args.hidden_dim).to(args.gpu).requires_grad_()
        nn.init.uniform_(
            tensor=ent_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        ent_embed_list.append(ent_embed)

        rel_freq = torch.zeros(nrelation)
        for r in data['train']['edge_type_ori'].reshape(-1):
            rel_freq[r] += 1
        rel_freq_list.append(rel_freq)

    rel_freq_mat = torch.stack(rel_freq_list).to(args.gpu)

    return train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
           rel_freq_mat, ent_embed_list, nrelation
