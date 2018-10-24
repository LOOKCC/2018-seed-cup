#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

import os
import torch
import argparse
import pickle
import torch.nn.functional as F
from torch import nn, optim
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score

from load_data import load_dataset
from models.Pool_xgb import Pool 

class CateManager(object):
    """
    """

    def __init__(self, args, LABEL):
        super(CateManager, self).__init__()
        with open(os.path.join(args.root, 'class_info.pkl'), 'rb') as fp:
            self.info = pickle.load(fp)
        self.vocabs = [L.vocab for L in LABEL]
        cate_num = [len(vocab) for vocab in self.vocabs]
        device = torch.device(args.device)
        self.cate1to2 = torch.zeros(cate_num[0], cate_num[1]).to(device)
        self.cate2to3 = torch.zeros(cate_num[1], cate_num[2]).to(device)
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

    def merge_weights(self, cate_out, label=None):
        if self.merge:
            # cate_out[1] = cate_out[1] * self.cate1to2[cate_out[0].max(1)[1]]
            # cate_out[2] = cate_out[2] * self.cate2to3[cate_out[1].max(1)[1]]
            # cate_out[1] = cate_out[1] * torch.mm(cate_out[0], self.cate1to2)
            # cate_out[2] = cate_out[2] * torch.mm(cate_out[1], self.cate2to3)
            # cate_out[1] = cate_out[1] * torch.mm(F.softmax(cate_out[0]), self.cate1to2)
            # cate_out[2] = cate_out[2] * torch.mm(F.softmax(cate_out[1]), self.cate2to3)
            if label is not None:
                cate_out[1] = cate_out[1] * (self.cate1to2[label[0]]*101-100)
                cate_out[2] = cate_out[2] * (self.cate2to3[label[1]]*101-100)
            else:
                cate_out[1] = cate_out[1] * \
                    (self.cate1to2[cate_out[0].max(1)[1]]*101-100)
                cate_out[2] = cate_out[2] * \
                    (self.cate2to3[cate_out[1].max(1)[1]]*101-100)
        return cate_out


def train(args, train_iter, TEXT, LABEL, ID, cate_manager, checkpoint=None):
    # get device
    device = torch.device(args.device)
    model = Pool(TEXT, LABEL, dropout=args.dropout,
                    freeze=args.freeze).to(device)

    parameters = [x for x in model.parameters() if x.requires_grad == True]
    # train
    save_list = []
    model = model.train()
    print('====   Generating..   ====')
    for iter_num, batch in enumerate(train_iter):
        label = (batch.cate1_id, batch.cate2_id, batch.cate3_id)
        output = model(
            torch.cat((batch.title_words, batch.disc_words), dim=1))
        for i in range(output.shape[0]):
            save_dict  = {'ID': batch.item_id[i], 'feature': output[i], 'cate1': label[0][i], 'cate2': label[1][i], 'cate3': label[2][i]}
            save_list.append(save_dict)
    f = opne(os.path.join(args.root, 'xgb.pkl'))
    pickle.dump(save_list,f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../data',
                        help='Path to dataset (default="../data")')
    parser.add_argument('--device', default='cuda:0',
                        help='Device to use (default="cuda:0")')
    parser.add_argument('--snapshot', default=None,
                        help='Path to save model to save (default="checkpoints/crnn.pth")')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size (default=128)')

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
    train(args, train_iter, TEXT, LABEL, ID, cate_manager, checkpoint=checkpoint)
    
