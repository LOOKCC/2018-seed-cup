import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from sklearn.metrics import f1_score

from .represent_layer import SwemCat, SwemHier
from utils.timer import timer

CATE1_CNT = 20
CATE2_CNT = 135
CATE3_CNT = 265

TRAIN_SAMPLES = 911256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Cate2Classifier(nn.Module):
    def __init__(self, input_size, args, hier=0, word2vec=None, mask1=None):
        super(Cate2Classifier, self).__init__()
        if hier:
            self.swem_layer = SwemHier(input_size, args.embedding_dim, word2vec)
        else:
            self.swem_layer = SwemCat(input_size, args.embedding_dim, word2vec)
        self.fc = nn.Linear(args.embedding_dim*4, args.h_d)
        self.clf = nn.Linear(args.h_d, CATE2_CNT)
        self.bn = nn.BatchNorm1d(args.h_d)
        self.mask1 = mask1

    def forward(self, title, desc, t_len, d_len, cate1):
        swem_vec = self.swem_layer(title, desc, t_len, d_len)
        h = self.bn(self.fc(swem_vec))
        h = F.relu(h)
        output = self.clf(h)
        for i in range(title.size(0)):
            output[i, self.mask1[cate1[i]]] = -100
        return output

@timer
def train_epoch(epoch, model, dataloader, criterion, optimizer, args):
    model.train()
    print('Epoch', epoch)
    total_loss = 0
    pred2 = []
    target2 = []
    for cnt, (title, desc, cate1, cate2, cate3, t_len, d_len) in enumerate(dataloader):
        title = title.to(device)
        desc = desc.to(device)
        title = title.to(device)
        desc = desc.to(device)
        cate1 = cate1.to(device)
        cate2 = cate2.to(device)
        optimizer.zero_grad()
        output = model(title, desc, t_len, d_len, cate1)
        loss = criterion(output, cate2)
        total_loss += loss.item()
        if (cnt + 1) % 100 == 0:
            print('{} / {} Training current loss {:.4}'.format((cnt + 1) * args.batch_size,
                                                             TRAIN_SAMPLES,
                                                             total_loss / (cnt + 1)))
        loss.backward()
        optimizer.step()
        pred2.extend(output.argmax(1).tolist())
        target2.extend(cate2.tolist())
    print('Training Total Loss {:.4}'.format(total_loss / (cnt + 1)))
    cate2_score = f1_score(target2, pred2, average='macro')
    print('Training cate2 f1 score: {:.4}'.format(cate2_score))


def eval_epoch(epoch, model, dataloader, best_score, optimizer, args):
    model.eval()
    pred2 = []
    target2 = []
    for title, desc, cate1, cate2, cate3, t_len, d_len in dataloader:
        title = title.to(device)
        desc = desc.to(device)
        cate1 = cate1.to(device)
        output = model(title, desc, t_len, d_len, cate1)
        pred2.extend(output.argmax(1).tolist())
        target2.extend(cate2.tolist())
    cate2_score = f1_score(target2, pred2, average='macro')
    print('Validation cate2 f1 score: {:.4}'.format(cate2_score))
    if cate2_score > best_score:
        print('==> Saving..')
        best_score = cate2_score
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'args': model.args,
                 'best_score': best_score,
                 'epoch': epoch}
        torch.save(state, os.path.join('checkpoint', args.ckpt))
    return  best_score