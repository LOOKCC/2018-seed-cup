# _*_ coding: utf-8 _*_

import os
import sys
import torch
import pickle
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors


def load_dataset(args):
    TEXT = data.Field(sequential=True, use_vocab=True,
                      tokenize=lambda x: x.split(','), batch_first=True)
    LABEL = [data.Field(sequential=True, tokenize=int, use_vocab=True,
                        unk_token=None) for _ in range(3)]
    ID = data.Field(sequential=True, use_vocab=False)

    datafields = [('item_id', None),  # 我们不会需要id，所以我们传入的filed是None
                  ('title_chars', TEXT), ('title_words', TEXT),
                  ('disc_chars', TEXT), ('disc_words', TEXT),
                  ('cate1_id', LABEL[0]), ('cate2_id', LABEL[1]),
                  ('cate3_id', LABEL[2])]

    train_data, valid_data = data.TabularDataset.splits(
        path=args.root,  # 数据存放的根目录
        root=args.root,  # 数据存放的根目录
        train='train_a.txt', validation='valid_a.txt',
        format='tsv',
        skip_header=True,  # 如果你的csv有表头, 确保这个表头不会作为数据处理
        fields=datafields)
    test_data = data.TabularDataset(
        path=os.path.join(args.root, 'test_a.txt'),
        format='tsv',
        skip_header=True,
        fields=datafields[:5])

    TEXT.build_vocab(train_data, valid_data, test_data, vectors=Vectors(
        'embedding/embedding.txt', cache='embedding/vec_cache/'))
    for L in LABEL:
        L.build_vocab(train_data)

    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.disc_words),
        sort=True,
        repeat=False,
        shuffle=True,
        sort_within_batch=False,
        device=args.device)

    test_iter = data.BucketIterator(
        test_data,
        batch_size=args.batch_size,
        sort=False,
        repeat=False,
        shuffle=False,
        sort_within_batch=False,
        train=False,
        device=args.device)

    cate_manager = CateManager(args, LABEL)

    return train_iter, valid_iter, test_iter, TEXT, LABEL, cate_manager


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
        for i in range(self.info):
            idx1 = self.vocabs[0].stoi[i]
            idx2 = [self.vocabs[1].stoi[j] for j in self.info[i].keys()]
            self.cate1to2[idx1, idx2] = 1
            for j in range(self.info[i]):
                idx2 = self.vocabs[1].stoi[j]
                idx3 = [self.vocabs[2].stoi[k] for k in self.info[i][j].keys()]
                self.cate2to3[idx2, idx3] = 1

    def merge_weights(self, cate_out):
        cate_out[1] *= torch.mm(cate_out[0], self.cate1to2)
        cate_out[2] *= torch.mm(cate_out[1], self.cate2to3)
        return cate_out
