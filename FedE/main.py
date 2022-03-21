# @Author   : Chen Mingyang
# @Time     : 2020/9/2
# @FileName : z_main.py


from dataloader import *
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import logging
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F
import argparse
import json
import pickle
from kge_model import KGEModel
from fede import FedE


class KGERunner():
    def __init__(self, args, data):
        self.args = args
        self.data = data

        if args.run_mode == 'Entire':
            train_dataset, valid_dataset, test_dataset, nrelation, nentity = get_task_dataset_entire(data, args)
        else:
            train_dataset, valid_dataset, test_dataset, nrelation, nentity = get_task_dataset(data, args)

        self.nentity = nentity
        self.nrelation = nrelation

        # embedding
        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        if args.model in ['RotatE', 'ComplEx']:
            self.entity_embedding = torch.zeros(self.nentity, args.hidden_dim * 2).to(args.gpu).requires_grad_()
        else:
            self.entity_embedding = torch.zeros(self.nentity, args.hidden_dim).to(args.gpu).requires_grad_()
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        if args.model in ['ComplEx']:
            self.relation_embedding = torch.zeros(self.nrelation, args.hidden_dim * 2).to(args.gpu).requires_grad_()
        else:
            self.relation_embedding = torch.zeros(self.nrelation, args.hidden_dim).to(args.gpu).requires_grad_()
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )

        # dataloader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle = True,
            collate_fn = TrainDataset.collate_fn
        )

        if args.run_mode == 'Entire':
            self.valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=args.test_batch_size,
                collate_fn=TestDataset_Entire.collate_fn
            )

            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.test_batch_size,
                collate_fn=TestDataset_Entire.collate_fn
            )
        else:
            self.valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=args.test_batch_size,
                collate_fn=TestDataset.collate_fn
            )

            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size = args.test_batch_size,
                collate_fn=TestDataset.collate_fn
            )

        # model
        self.kge_model = KGEModel(args, args.model)

        self.optimizer = torch.optim.Adam(
            [{'params': self.entity_embedding},
             {'params': self.relation_embedding}], lr=args.lr
        )

    def before_test_load(self):
        state = torch.load(os.path.join(self.args.state_dir, self.args.name + '.best'),
                           map_location=self.args.gpu)
        self.relation_embedding = state['rel_emb']
        self.entity_embedding = state['ent_emb']

    # def load_from_multi(self):
    #     state = torch.load(os.path.join(self.args.state_dir, self.args.name + '.best'),
    #                        map_location=self.args.gpu)
    #     self.relation_embedding = state['rel_emb']
    #     self.entity_embedding = state['ent_emb']
    #
    #     nentity = len(np.unique(data['train']['edge_index'].reshape(-1)))
    #     nrelation = len(np.unique(data['train']['edge_type'].reshape(-1)))
    #     rel_purm = np.zeros(nrelation, dtype=np.int64)
    #     ent_purm = np.zeros(nentity, dtype=np.int64)
    #     for i in range(data['train']['edge_index'].shape[1]):
    #         h, r, t = data['train']['edge_index'][0][i], data['train']['edge_type'][i], \
    #                   data['train']['edge_index'][1][i]
    #         h_ori, r_ori, t_ori = data['train']['edge_index_ori'][0][i], data['train']['edge_type_ori'][i], \
    #                               data['train']['edge_index_ori'][1][i]
    #         ent_purm[h] = h_ori
    #         rel_purm[r] = r_ori
    #         ent_purm[t] = t_ori
    #     ent_purm = torch.LongTensor(ent_purm)
    #     rel_purm = torch.LongTensor(rel_purm)
    #
    #     self.relation_embedding = self.relation_embedding[rel_purm]
    #     self.entity_embedding = self.entity_embedding[ent_purm]

    def write_training_loss(self, loss, e):
        self.args.writer.add_scalar("training/loss", loss, e)

    def write_evaluation_result(self, results, e):
        self.args.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.args.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.args.writer.add_scalar("evaluation/hits3", results['hits@3'], e)
        self.args.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def save_checkpoint(self, e):
        state = {'rel_emb': self.relation_embedding,
                 'ent_emb': self.entity_embedding}
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

    def train(self):
        best_epoch = 0
        best_mrr = 0
        bad_count = 0

        for epoch in range(self.args.max_epoch):
            losses = []
            self.kge_model.train()
            for batch in self.train_dataloader:

                positive_sample, negative_sample, subsampling_weight = batch

                positive_sample = positive_sample.to(args.gpu)
                negative_sample = negative_sample.to(args.gpu)
                subsampling_weight = subsampling_weight.to(args.gpu)

                negative_score = self.kge_model((positive_sample, negative_sample),
                                                  self.relation_embedding,
                                                  self.entity_embedding)

                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)

                positive_score = self.kge_model(positive_sample,
                                                self.relation_embedding, self.entity_embedding, neg=False)

                positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()

                loss = (positive_sample_loss + negative_sample_loss) / 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            if epoch % self.args.log_per_epoch == 0:
                logging.info('epoch: {} | loss: {:.4f}'.format(epoch, np.mean(losses)))
                self.write_training_loss(np.mean(losses), epoch)

            if epoch % self.args.check_per_epoch == 0:
                if args.run_mode == 'Entire':
                    eval_res = self.evaluate_multi()
                else:
                    eval_res = self.evaluate()
                self.write_evaluation_result(eval_res, epoch)

                if eval_res['mrr'] > best_mrr:
                    best_mrr = eval_res['mrr']
                    best_epoch = epoch
                    logging.info('best model | mrr {:.4f}'.format(best_mrr))
                    self.save_checkpoint(epoch)
                    bad_count = 0
                else:
                    bad_count += 1
                    logging.info('best model is at round {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_epoch, best_mrr, bad_count))

            if bad_count >= self.args.early_stop_patience:
                logging.info('early stop at round {}'.format(epoch))
                break

        logging.info('finish training')
        logging.info('save best model')
        self.save_model(best_epoch)

        logging.info('eval on test set')
        self.before_test_load()
        if args.run_mode == 'multi_client_train':
            eval_res = self.evaluate_multi(eval_split='test')
        else:
            eval_res = self.evaluate(eval_split='test')

    def evaluate_multi(self, eval_split='valid'):

        if eval_split == 'test':
            dataloader = self.test_dataloader
        elif eval_split == 'valid':
            dataloader = self.valid_dataloader

        client_ranks = ddict(list)
        all_ranks = []
        for batch in dataloader:

            triplets, labels, triple_idx = batch
            triplets, labels = triplets.to(args.gpu), labels.to(args.gpu)
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            pred = self.kge_model((triplets, None),
                                   self.relation_embedding,
                                   self.entity_embedding)
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]

            ranks = ranks.float()

            for i in range(args.num_multi):
                client_ranks[i].extend(ranks[triple_idx == i].tolist())

            all_ranks.extend(ranks.tolist())

        for i in range(args.num_multi):
            results = ddict(float)
            ranks = torch.tensor(client_ranks[i])
            count = torch.numel(ranks)
            results['count'] = count
            results['mr'] = torch.sum(ranks).item() / count
            results['mrr'] = torch.sum(1.0 / ranks).item() / count
            for k in [1, 3, 10]:
                results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k]) / count
            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
                results['mrr'], results['hits@1'],
                results['hits@3'], results['hits@10']))

        results = ddict(float)
        ranks = torch.tensor(all_ranks)
        count = torch.numel(ranks)
        results['count'] = count
        results['mr'] = torch.sum(ranks).item() / count
        results['mrr'] = torch.sum(1.0 / ranks).item() / count
        for k in [1, 3, 10]:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k]) / count
        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@3'], results['hits@10']))

        return results

    def evaluate(self, eval_split='valid'):
        results = ddict(float)

        if eval_split == 'test':
            dataloader = self.test_dataloader
        elif eval_split == 'valid':
            dataloader = self.valid_dataloader

        pred_list = []
        rank_list = []
        results_list = []
        for batch in dataloader:
            triplets, labels = batch
            triplets, labels = triplets.to(args.gpu), labels.to(args.gpu)
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            pred = self.kge_model((triplets, None),
                                  self.relation_embedding,
                                  self.entity_embedding)
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred

            pred_argsort = torch.argsort(pred, dim=1, descending=True)
            ranks = 1 + torch.argsort(pred_argsort, dim=1, descending=False)[b_range, tail_idx]

            pred_list.append(pred_argsort[:, :10])
            rank_list.append(ranks)

            ranks = ranks.float()

            for idx, tri in enumerate(triplets):
                results_list.append([tri.tolist(), ranks[idx].item()])

            count = torch.numel(ranks)
            results['count'] += count
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [1, 3, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        torch.save(torch.cat(pred_list, dim=0), os.path.join(args.state_dir,
                                                             args.name + '_' + str(args.one_client_idx) + '.pred'))
        torch.save(torch.cat(rank_list), os.path.join(args.state_dir,
                                                      args.name + '_' + str(args.one_client_idx) + '.rank'))

        for k, v in results.items():
            if k != 'count':
                results[k] /= results['count']

        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@3'], results['hits@10']))

        test_rst_file = os.path.join(args.log_dir, args.name + '.test.rst')
        pickle.dump(results_list, open(test_rst_file, 'wb'))

        return results


def test_pretrain(args, all_data):
    data_len = len(all_data)
    #
    # train_dataloader_list, valid_dataloader_list, test_dataloader_list, ent_emb_list, rel_update_weights, g_list \
    #     = get_all_clients(all_data, args)
    #
    # total_test_data_size = sum([len(test_dataloader_list[i].dataset) for i in range(data_len)])
    # eval_weights = [len(test_dataloader_list[i].dataset) / total_test_data_size for i in range(data_len)]

    embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
    kge_model = KGEModel(args, model_name=args.model)

    # rel_result = ddict(list)
    # rel_result_bydata = ddict(lambda : ddict(list))
    results = ddict(float)
    for i, data in enumerate(all_data):
        one_results = ddict(float)
        state = torch.load('../LTLE/fed_state/fb15k237_fed10_client_{}.best'.format(i), map_location=args.gpu)
        rel_embed = state['rel_emb'].detach()
        ent_embed = state['ent_emb'].detach()

        train_dataset, valid_dataset, test_dataset, nrelation, nentity = get_task_dataset(data, args)
        test_dataloader_tail = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            # num_workers=max(1, args.num_cpu),
            collate_fn=TestDataset.collate_fn
        )

        client_res = ddict(float)
        for batch in test_dataloader_tail:
            triplets, labels, mode = batch
            # triplets, labels, mode = next(test_dataloader_list[i].__iter__())
            triplets, labels = triplets.to(args.gpu), labels.to(args.gpu)
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            pred = kge_model((triplets, None),
                              rel_embed,
                              ent_embed,
                              mode=mode)
            b_range = torch.arange(pred.size()[0], device=args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]

            ranks = ranks.float()
            count = torch.numel(ranks)

            results['count'] += count
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            one_results['count'] += count
            one_results['mr'] += torch.sum(ranks).item()
            one_results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [1, 3, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])
                one_results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        for k, v in one_results.items():
            if k != 'count':
                one_results[k] = v / one_results['count']

        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
            one_results['mrr'], one_results['hits@1'],
            one_results['hits@3'], one_results['hits@10']))

    for k, v in results.items():
        if k != 'count':
            results[k] = v / results['count']

    logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@3: {:.4f}, hits@10: {:.4f}'.format(
        results['mrr'], results['hits@1'],
        results['hits@3'], results['hits@10']))

    return results


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='FB15K237-Fed3.pkl', type=str)
    parser.add_argument('--name', default='fb15k237_fed3_TransE', type=str)
    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)
    parser.add_argument('--run_mode', default='FedE', choices=['FedE',
                                                                        'Single',
                                                                        'Entire',
                                                                        'test_pretrain',
                                                                        'get_valid_score'])
    parser.add_argument('--num_multi', default=3, type=int)

    parser.add_argument('--model', default='TransE', choices=['TransE', 'RotatE', 'DistMult', 'ComplEx'])

    # one task hyperparam
    parser.add_argument('--one_client_idx', default=0, type=int)
    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--log_per_epoch', default=1, type=int)
    parser.add_argument('--check_per_epoch', default=10, type=int)


    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--num_neg', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=int)

    # for FedE
    parser.add_argument('--num_client', default=3, type=int)
    parser.add_argument('--max_round', default=10000, type=int)
    parser.add_argument('--local_epoch', default=3)
    parser.add_argument('--fraction', default=1, type=float)
    parser.add_argument('--log_per_round', default=1, type=int)
    parser.add_argument('--check_per_round', default=5, type=int)

    parser.add_argument('--early_stop_patience', default=5, type=int)
    parser.add_argument('--gamma', default=10.0, type=float)
    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--hidden_dim', default=128, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_cpu', default=10, type=int)
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)

    # parser.add_argument('--negative_adversarial_sampling', default=True, type=bool)
    parser.add_argument('--seed', default=12345, type=int)

    args = parser.parse_args()
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

    if args.run_mode == 'FedE':
        all_data = pickle.load(open(args.data_path, 'rb'))
        learner = FedE(args, all_data)
        learner.train()
    elif args.run_mode == 'Single':
        all_data = pickle.load(open(args.data_path, 'rb'))
        data = all_data[args.one_client_idx]
        learner = KGERunner(args, data)
        learner.train()
    elif args.run_mode == 'Entire':
        all_data = pickle.load(open(args.data_path, 'rb'))
        learner = KGERunner(args, all_data)
        learner.train()
    # elif args.run_mode == 'test_pretrain':
    #     all_data = pickle.load(open(args.data_path, 'rb'))
    #     test_pretrain(args, all_data)
    # elif args.run_mode == 'get_valid_score':
    #     all_data = pickle.load(open(args.data_path, 'rb'))
    #     train_fusion(args, all_data, 3, 'nell995_fed3_client_{}_rotate.best', 'nell995_fed3_fed_rotate.best')

