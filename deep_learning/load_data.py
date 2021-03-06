# _*_ coding: utf-8 _*_

import os
import sys
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors


def tokenize(x):
    x = x.split(',')
    random.shuffle(x)
    return x


def load_dataset(args):
    TEXT = data.Field(sequential=True, use_vocab=True,
                      tokenize=tokenize, batch_first=True)
    LABEL = [data.Field(sequential=False, use_vocab=True,
                        unk_token=None) for _ in range(3)]
    ID = data.Field(sequential=False, use_vocab=True)

    datafields = [('item_id', ID),  # 我们不会需要id，所以我们传入的filed是None
                  ('title_chars', TEXT), ('title_words', TEXT),
                  ('disc_chars', TEXT), ('disc_words', TEXT),
                  ('cate1_id', LABEL[0]), ('cate2_id', LABEL[1]),
                  ('cate3_id', LABEL[2])]

    train_data, valid_data = data.TabularDataset.splits(
        path=args.root,  # 数据存放的根目录
        root=args.root,  # 数据存放的根目录
        train='train_b.txt', validation='valid_b.txt',
        format='tsv',
        skip_header=True,  # 如果你的csv有表头, 确保这个表头不会作为数据处理
        fields=datafields)
    test_data = data.TabularDataset(
        path=os.path.join(args.root, 'test_b.txt'),
        format='tsv',
        skip_header=True,
        fields=datafields[:5])

    if not os.path.exists('embedding/vec_cache'):
        os.makedirs('embedding/vec_cache')
    TEXT.build_vocab(train_data, valid_data, test_data, vectors=Vectors(
        'embedding/embedding_256.txt', cache='embedding/vec_cache/'))
    for L in LABEL:
        L.build_vocab(train_data, valid_data)
    ID.build_vocab(test_data)

    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.disc_words),
        sort=False,
        repeat=False,
        shuffle=True,
        sort_within_batch=False,
        device=args.device)

    test_iter = data.BucketIterator(
        test_data,
        batch_size=args.batch_size,
        sort_key=lambda x: len(x.disc_words),
        sort=False,
        repeat=False,
        shuffle=False,
        sort_within_batch=False,
        train=False,
        device=args.device)

    return train_iter, valid_iter, test_iter, TEXT, LABEL, ID
