#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

import os
import torch
import argparse
from torch import nn, optim
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score

from load_data import load_dataset
from models.model import TextCNN


def train(args, train_iter, TEXT, LABEL, cate_manager, checkpoint=None):
    model = TextCNN(TEXT, LABEL)
    criterion = [nn.CrossEntropyLoss() for _ in range(len(LABEL))]
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch']

    # get device
    device = torch.device(args.device)
    model = model.to(device)
    criterion = criterion.to(device)

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
            output = model(batch.title_words)
            output = cate_manager.merge_weights(output)
            for i in range(len(LABEL)):
                loss = criterion[i](output[i], label[i])
                loss.backward()
                loss_sum += loss.item()
                all_pred[i].extend(output[i].max(1)[1].tolist())
                all_label[i].extend(label[i].tolist())
            optimizer.step()
        print('Loss = {}  \ttime: {}'.format(loss_sum/(iter_num+1),
                                             datetime.now()-start_time))
        print(*['Cate{} F1 score: {}  \t'.format(i, f1_score(all_label[i],
                                            all_pred[i], average='weighted'))])
    if args.snapshot_path is None:
        snapshot_path = 'snapshot/model_{}.pth'.format(
            start_epoch+args.check_epoch-1)
    checkpoint = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'epoch': start_epoch+args.check_epoch
    }
    torch.save(checkpoint, snapshot_path)
    print('Model saved in {}'.format(snapshot_path))


@torch.no_grad()
def evaluate(args, valid_iter, TEXT, LABEL, cate_manager, checkpoint):
    model = TextCNN(TEXT, LABEL)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    # get device
    device = torch.device(args.device)
    model = model.to(device)

    # evaluate
    model = model.eval()
    print('====   Validing..   ====')
    start_time = datetime.now()
    all_pred, all_label = [[], [], []], [[], [], []]
    for iter_num, batch in enumerate(valid_iter):
        output = model(batch.title_words)
        for i in range(len(output)):
            all_pred[i].extend(output[i].max(1)[1].tolist())
            all_label[i].extend(label[i].tolist())
    print('time: {}'.format(datetime.now()-start_time))
    print(*['Cate{} F1 score: {}  \t'.format(i, f1_score(all_label[i],
                                        all_pred[i], average='weighted'))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../data',
                        help='Path to dataset (default="../data")')
    parser.add_argument('--device', default='cpu',
                        help='Device to use (default="cpu")')
    parser.add_argument('--snapshot', default=None,
                        help='Path to save model to save (default="checkpoints/crnn.pth")')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size (default=128)')

    parser.add_argument('--snapshot_path', default=None,
                        help='Path to save model (default="snapshot/model_{epoch}.pth")')
    parser.add_argument('--epoch_num', type=int, default=50,
                        help='Number of epochs to train for (default=50)')
    parser.add_argument('--check_epoch', type=int, default=10,
                        help='Epoch to save and test (default=10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Optimizer (default=0.001)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for Optimizer (default=0)')

    parser.add_argument('--valid', action='store_true',
                        help='Evaluate only once')
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

    train_iter, valid_iter, test_iter, TEXT, LABEL, cate_manager = load_dataset(args)
    for i in range(args.epoch_num//args.check_epoch):
        if not args.valid:
            train(args, train_iter, TEXT, LABEL,
                  cate_manager, checkpoint=checkpoint)
            evaluate(args, valid_iter, TEXT, LABEL, cate_manager, checkpoint)
        else:
            evaluate(args, valid_iter, TEXT, LABEL, cate_manager, checkpoint)
            break
