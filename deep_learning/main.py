#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

import os
import torch
import argparse
import pickle
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score

from load_data import load_dataset
from models.model import Model


class CateManager(object):
    """
    """

    def __init__(self, args, LABEL):
        super(CateManager, self).__init__()
        with open(os.path.join(args.root, 'class_info.pkl'), 'rb') as fp:
            self.info = pickle.load(fp)
        self.vocabs = [L.vocab for L in LABEL]
        cate_num = [len(vocab) for vocab in self.vocabs]
        self.device = torch.device(args.device)
        self.cate1to2 = torch.zeros(cate_num[0], cate_num[1]).to(self.device)
        self.cate2to3 = torch.zeros(cate_num[1], cate_num[2]).to(self.device)
        self.merge = args.merge
        for i in self.info:
            idx1 = self.vocabs[0].stoi[str(i)]
            idx2 = [self.vocabs[1].stoi[str(j)] for j in self.info[i].keys()]
            self.cate1to2[idx1, idx2] = 1
            for j in self.info[i]:
                idx2 = self.vocabs[1].stoi[str(j)]
                idx3 = [self.vocabs[2].stoi[str(k)]
                        for k in self.info[i][j].keys()]
                self.cate2to3[idx2, idx3] = 1

    def get_weights(self, LABEL):
        weights = []
        for i in range(len(LABEL)):
            freqs = LABEL[i].vocab.freqs
            itos = LABEL[i].vocab.itos
            if i == 0:
                num = sum(freqs.values())
                weights.append(torch.Tensor(
                    [num/freqs[itos[j]]/len(freqs) for j in range(len(freqs))]).to(self.device))
            elif i == 1:
                weights.append(torch.zeros(len(freqs)).to(self.device))
                freq_tensor = torch.Tensor(
                    sorted(freqs.values(), reverse=True)).to(self.device)
                for j in range(len(LABEL[i-1].vocab.freqs)):
                    num = LABEL[i-1].vocab.freqs[LABEL[i-1].vocab.itos[j]]
                    weights[-1] += self.cate1to2[j] / \
                        freq_tensor * num / sum(self.cate1to2[j]).item()
            else:
                weights.append(torch.zeros(len(freqs)).to(self.device))
                freq_tensor = torch.Tensor(
                    sorted(freqs.values(), reverse=True)).to(self.device)
                for j in range(len(LABEL[i-1].vocab.freqs)):
                    num = LABEL[i-1].vocab.freqs[LABEL[i-1].vocab.itos[j]]
                    weights[-1] += self.cate2to3[j] / \
                        freq_tensor * num / sum(self.cate2to3[j]).item()
        return weights

    def merge_weights(self, cate_out, label=None):
        if self.merge:
            # cate_out[1] = cate_out[1] * self.cate1to2[cate_out[0].max(1)[1]]
            # cate_out[2] = cate_out[2] * self.cate2to3[cate_out[1].max(1)[1]]
            # cate_out[1] = cate_out[1] * torch.mm(cate_out[0], self.cate1to2)
            # cate_out[2] = cate_out[2] * torch.mm(cate_out[1], self.cate2to3)
            # cate_out[1] = cate_out[1] * torch.mm(F.softmax(cate_out[0]), self.cate1to2)
            # cate_out[2] = cate_out[2] * torch.mm(F.softmax(cate_out[1]), self.cate2to3)
            if label is not None:
                cate_out[1] = cate_out[1] * (self.cate1to2[label[0]]*2-1)
                cate_out[2] = cate_out[2] * (self.cate2to3[label[1]]*2-1)
                # cate_out[1][np.where(self.cate1to2[label[0]] == 0)] = -100
                # cate_out[2][np.where(self.cate2to3[label[1]] == 0)] = -100
            else:
                cate_out[1] = cate_out[1] * \
                    (self.cate1to2[cate_out[0].max(1)[1]]*2-1)
                cate_out[2] = cate_out[2] * \
                    (self.cate2to3[cate_out[1].max(1)[1]]*2-1)
                # cate_out[1][np.where(
                #     self.cate1to2[cate_out[0].max(1)[1]] == 0)] = -100
                # cate_out[2][np.where(
                #     self.cate2to3[cate_out[1].max(1)[1]] == 0)] = -100
        return cate_out


def train(args, train_iter, TEXT, LABEL, cate_manager, checkpoint=None):
    # get device
    device = torch.device(args.device)
    model = Model(TEXT, LABEL, dropout=args.dropout,
                  freeze=args.freeze).to(device)
    start_epoch = 0

    parameters = [x for x in model.parameters() if x.requires_grad == True]
    optimizer = optim.Adam(parameters, lr=args.lr,
                           weight_decay=args.weight_decay)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch']

    weights = cate_manager.get_weights(LABEL)

    # train
    model = model.train()
    print('====   Training..   ====')
    for epoch in range(start_epoch, start_epoch+args.check_epoch):
        print('----    Epoch: %d    ----' % (epoch, ))
        loss_sum = 0
        all_pred, all_label = [[], [], []], [[], [], []]
        start_time = datetime.now()
        for iter_num, batch in enumerate(train_iter):
            label = (batch.cate1_id, batch.cate2_id, batch.cate3_id)
            optimizer.zero_grad()
            output, result = model(batch)
            output = [cate_manager.merge_weights(x, label) for x in output]
            result = cate_manager.merge_weights(result)
            for i in range(len(LABEL)):
                for j in range(len(output)):
                    loss = F.cross_entropy(
                        output[j][i], label[i], weight=weights[i])
                    loss.backward(retain_graph=True)
                    loss_sum += loss.item()
                all_pred[i].extend(result[i].max(1)[1].tolist())
                all_label[i].extend(label[i].tolist())
            optimizer.step()
        print('Loss = {}  \ttime: {}'.format(loss_sum/(iter_num+1),
                                             datetime.now()-start_time))
        print(*['Cate{} F1 score: {}  \t'.format(i+1, f1_score(all_label[i],
                                                               all_pred[i], average='macro')) for i in range(len(LABEL))])
        if args.snapshot_path is None:
            snapshot_path = 'snapshot/model_{}.pth'.format(epoch)
        if not os.path.exists(os.path.dirname(args.snapshot_path)):
            os.makedirs(os.path.dirname(args.snapshot_path))
        checkpoint = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch+1
        }
        torch.save(checkpoint, snapshot_path)
        print('Model saved in {}'.format(snapshot_path))
    return checkpoint


# @torch.no_grad()
def evaluate(args, valid_iter, TEXT, LABEL, cate_manager, checkpoint):
    # get device
    device = torch.device(args.device)
    model = Model(TEXT, LABEL, dropout=args.dropout,
                  freeze=args.freeze).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    # evaluate
    model = model.eval()
    print('====   Validing..   ====')
    start_time = datetime.now()
    all_pred, all_label = [[], [], []], [[], [], []]
    for iter_num, batch in enumerate(valid_iter):
        label = (batch.cate1_id, batch.cate2_id, batch.cate3_id)
        output, result = model(batch, training=False)
        result = cate_manager.merge_weights(result)
        for i in range(len(result)):
            all_pred[i].extend(result[i].max(1)[1].tolist())
            all_label[i].extend(label[i].tolist())
    print('time: {}'.format(datetime.now()-start_time))
    print(*['Cate{} F1 score: {}  \t'.format(i+1, f1_score(all_label[i],
                                                           all_pred[i], average='macro')) for i in range(len(LABEL))])


# @torch.no_grad()
def test(args, test_iter, TEXT, LABEL, ID, cate_manager, checkpoint):
    # get device
    device = torch.device(args.device)
    model = Model(TEXT, LABEL, dropout=args.dropout,
                  freeze=args.freeze).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    # evaluate
    model = model.eval()
    print('====    Testing..   ====')
    start_time = datetime.now()
    all_pred, ids = [[], [], []], []
    for iter_num, batch in enumerate(test_iter):
        ids.extend(batch.item_id.tolist())
        output, result = model(batch, training=False)
        result = cate_manager.merge_weights(result)
        for i in range(len(result)):
            all_pred[i].extend(result[i].max(1)[1].tolist())
    print('time: {}'.format(datetime.now()-start_time))
    with open('../data/out.txt', 'w') as fp:
        fp.write('item_id\tcate1_id\tcate2_id\tcate3_id\n')
        for i in range(len(all_pred[0])):
            fp.write(ID.vocab.itos[ids[i]]+'\t')
            fp.write(
                '\t'.join([LABEL[j].vocab.itos[all_pred[j][i]] for j in range(3)]))
            fp.write('\n')
    print('Result saved in ../../data/out.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../data',
                        help='Path to dataset (default="../data")')
    parser.add_argument('--device', default='cuda:0',
                        help='Device to use (default="cuda:0")')
    parser.add_argument('--snapshot', default=None,
                        help='Path to save model to save (default="checkpoints/crnn.pth")')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Input batch size (default=64)')

    parser.add_argument('--snapshot_path', default=None,
                        help='Path to save model (default="snapshot/model_{epoch}.pth")')
    parser.add_argument('--epoch_num', type=int, default=50,
                        help='Number of epochs to train for (default=50)')
    parser.add_argument('--check_epoch', type=int, default=5,
                        help='Epoch to save and test (default=5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Optimizer (default=0.001)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for Optimizer (default=0)')

    parser.add_argument('--valid', action='store_true',
                        help='Evaluate only once')
    parser.add_argument('--test', action='store_true',
                        help='Test only once')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze embedding layer or not')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Rate of dropout (default=0)')
    parser.add_argument('--merge', action='store_true',
                        help='Merge probality of every level')
    args = parser.parse_args()
    print(args)

    # load models, optimizer, start_iter
    checkpoint = None

    if args.valid and args.snapshot is None:
        print('Please set the "snapshot" argument!')
        exit(0)

    if args.snapshot is not None and os.path.exists(args.snapshot):
        print('Pre-trained model detected.\nLoading model...')
        checkpoint = torch.load(args.snapshot)

    train_iter, valid_iter, test_iter, TEXT, LABEL, ID = load_dataset(args)
    cate_manager = CateManager(args, LABEL)
    for i in range(args.epoch_num//args.check_epoch):
        if not args.valid and not args.test:
            checkpoint = train(args, train_iter, TEXT, LABEL,
                               cate_manager, checkpoint=checkpoint)
            evaluate(args, valid_iter, TEXT, LABEL, cate_manager, checkpoint)
        elif args.valid:
            evaluate(args, valid_iter, TEXT, LABEL, cate_manager, checkpoint)
            break
        else:
            test(args, test_iter, TEXT, LABEL, ID, cate_manager, checkpoint)
            break
