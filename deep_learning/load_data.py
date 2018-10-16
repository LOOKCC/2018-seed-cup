# _*_ coding: utf-8 _*_

import os
import sys
import torch
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors

def load_dataset(args):
    tokenize = lambda x: x.split(',')
    TEXT = data.Field(sequential=True, tokenize=tokenize, batch_first=True)
    LABEL = [data.Field(sequential=False, tensor_type=torch.FloatTensor) for _ in range(3)]
    ID = data.Field(sequential=True, use_vocab=False)

    datafields = [('item_id', None), # 我们不会需要id，所以我们传入的filed是None
                 ('title_chars', TEXT), ('title_words', TEXT),
                 ('disc_chars', TEXT), ('disc_words', TEXT),
                 ('cate1_id', LABEL[0]), ('cate2_id', LABEL[1]),
                 ('cate3_id', LABEL[2])]

    train_data, valid_data = data.TabularDataset.splits(
                               path='../data/', # 数据存放的根目录
                               root='../data/', # 数据存放的根目录
                               train='train_a.txt', validation='valid_a.txt',
                               format='tsv',
                               skip_header=True, # 如果你的csv有表头, 确保这个表头不会作为数据处理
                               fields=datafields)
    test_data = data.TabularDataset(path='../data/test_a.txt', format='tsv', skip_header=True, fields=datafields[:5])
    
    TEXT.build_vocab(train_data, valid_data, test_data, vectors=Vectors('embedding/embedding.txt', cache='embedding/vec_cache/'))
    map(lambda x:x.build_vocab(train_data), LABEL)

    print ('Length of Text Vocabulary: ' + str(len(TEXT.vocab)))
    print ('Vector size of Text Vocabulary: ', TEXT.vocab.vectors.size())
    print ('Label Length: ' + str(len(LABEL.vocab)))

    train_iter = data.BucketIterator(train_data, batch_size=args.batch_size, sort_key=lambda x: len(x.discription_words), sort=True, repeat=False, shuffle=True, sort_within_batch=False, device=args.device)

    return train_iter
