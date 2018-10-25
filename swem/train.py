import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import pickle
import argparse

from utils.dataset import TrainDataset, padding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_feature_path = './preproc/train_words.pkl'
valid_feature_path = './preproc/valid_words.pkl'
train_cate_path = './preproc/train_cate.pkl'
valid_cate_path = './preproc/valid_cate.pkl'

WORDS_CNT = 72548


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=int, default=1,
                        help='train cate m classifier, default=1')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size, default=64')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate, default=1e-3')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='l2 weight decay, default=0')
    # parser.add_argument('-d', '--drop', action='store_true', help='drop some words in docs while training')
    # parser.add_argument('--drop_rate', type=float, default=0.2, help='rate of dropped words, default=0.2')
    parser.add_argument('--h_d', type=int, default=128,
                        help='hidden dim, default=128')
    parser.add_argument('--ckpt', required=True,
                        help='load/save checkpoint path')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='word/char embedding dim, default=512')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_cmd()

    best_score = 0
    start_epoch = 0

    if args.model == 1:
        from models.cate1_classifier import *
        if args.resume:
            state = torch.load(args.ckpt)
            model = Cate1Classifier(WORDS_CNT+1, state['args'])
            model.load_state_dict(state['model'])
            start_epoch = state['epoch'] + 1
            best_score = state['best_score']
        else:
            word2vec = torch.load('./preproc/word2vec.pth')
            model = Cate1Classifier(WORDS_CNT+1, args, word2vec)

    elif args.model == 2:
        from models.cate2_classifier import *
        if args.resume:
            state = torch.load(args.ckpt)
            with open('./preproc/mask.pkl', 'rb') as fp:
                mask1, mask2 = pickle.load(fp)
            model = Cate2Classifier(WORDS_CNT+1, state['args'], mask1=mask1)
            model.load_state_dict(state['model'])
            start_epoch = state['epoch'] + 1
            best_score = state['best_score']
        else:
            word2vec = torch.load('./preproc/word2vec.pth')
            with open('./preproc/mask.pkl', 'rb') as fp:
                mask1, mask2 = pickle.load(fp)
            model = Cate2Classifier(WORDS_CNT+1, args, word2vec, mask1)

    elif args.model == 3:
        from models.cate3_classifier import *
        if args.resume:
            state = torch.load(args.ckpt)
            with open('./preproc/mask.pkl', 'rb') as fp:
                mask1, mask2 = pickle.load(fp)
            model = Cate3Classifier(WORDS_CNT+1, state['args'], mask2=mask2)
            model.load_state_dict(state['model'])
            start_epoch = state['epoch'] + 1
            best_score = state['best_score']
        else:
            word2vec = torch.load('./preproc/word2vec.pth')
            with open('./preproc/mask.pkl', 'rb') as fp:
                mask1, mask2 = pickle.load(fp)
            model = Cate3Classifier(WORDS_CNT+1, args, word2vec, mask2)

    else:
        raise Exception

    model = model.to(device)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(model['optimizer'])

    with open(train_feature_path, 'rb') as fp:
        train_features = pickle.load(fp)

    train_title = [feature[0] for feature in train_features]
    train_desc = [feature[1] for feature in train_features]
    train_t_len = [len(feature[0]) for feature in train_features]
    train_d_len = [len(feature[1]) for feature in train_features]

    with open(train_cate_path, 'rb') as fp:
        train_cate = pickle.load(fp)
    train_cate1 = [cate[0] for cate in train_cate]
    train_cate2 = [cate[1] for cate in train_cate]
    train_cate3 = [cate[2] for cate in train_cate]

    train_title = padding(train_title, max(train_t_len))
    train_desc = padding(train_desc, max(train_d_len))

    train_t_len = torch.tensor(train_t_len)
    train_d_len = torch.tensor(train_d_len)

    train_cate1 = torch.tensor(train_cate1)
    train_cate2 = torch.tensor(train_cate2)
    train_cate3 = torch.tensor(train_cate3)

    train_set = TrainDataset(train_title, train_desc,
                          train_cate1, train_cate2, train_cate3,
                          train_t_len, train_d_len)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    with open(valid_feature_path, 'rb') as fp:
        valid_features = pickle.load(fp)
    valid_title = [feature[0] for feature in valid_features]
    valid_desc = [feature[1] for feature in valid_features]
    valid_t_len = [len(feature[0]) for feature in valid_features]
    valid_d_len = [len(feature[1]) for feature in valid_features]

    with open(valid_cate_path, 'rb') as fp:
        valid_cate = pickle.load(fp)
    valid_cate1 = [cate[0] for cate in valid_cate]
    valid_cate2 = [cate[1] for cate in valid_cate]
    valid_cate3 = [cate[2] for cate in valid_cate]

    valid_title = padding(valid_title, max(valid_t_len))
    valid_desc = padding(valid_desc, max(valid_d_len))
    valid_t_len = torch.tensor(valid_t_len)
    valid_d_len = torch.tensor(valid_d_len)

    valid_cate1 = torch.tensor(valid_cate1)
    valid_cate2 = torch.tensor(valid_cate2)
    valid_cate3 = torch.tensor(valid_cate3)

    valid_set = TrainDataset(valid_title, valid_desc,
                          valid_cate1, valid_cate2, valid_cate3,
                          valid_t_len, valid_d_len)
    valid_loader = data.DataLoader(valid_set, batch_size=args.batch_size, num_workers=4)

    for epoch in range(start_epoch, start_epoch+20):
        train_epoch(epoch, model, train_loader, criterion, optimizer, args)
        with torch.no_grad():
            best_score = eval_epoch(epoch, model, valid_loader, best_score, args)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

