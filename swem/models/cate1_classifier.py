import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

from .represent_layer import SwemCat

WORDS_CNT = 34835
CHARS_CNT = 3939

CATE1_CNT = 10
CATE2_CNT = 64
CATE3_CNT = 125

TRAIN_SAMPLES = 140562

embedding_dim = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Cate1Classifier(nn.Module):
    def __init__(self, args, word2vec=None):
        super(Cate1Classifier, self).__init__()
        self.swem_layer = SwemCat(args, word2vec)
        self.args = args
        if args.h_d > 0:
            self.fc = nn.Linear(embedding_dim*4, args.h_d)
            self.clf = nn.Linear(args.h_d, CATE1_CNT)
            self.bn = nn.BatchNorm1d(args.h_d)
        else:
            self.clf = nn.Linear(embedding_dim*4, CATE1_CNT)

    def forward(self, title, desc, t_len, d_len, mode=1):
        batch_size = title.size(0)
        swem_vec = torch.zeros((batch_size, embedding_dim*4), device=device)
        for i in range(batch_size):
            swem_vec[i] = self.swem_layer(title[i], desc[i], t_len[i], d_len[i], mode)
        if self.args.h_d > 0:
            h = self.bn(self.fc(swem_vec))
            h = F.relu(h)
            output = self.clf(h)
        else:
            output = self.clf(swem_vec)
        return output


def train_epoch(epoch, model, dataloader, criterion, optimizer, args):
    model.train()
    print('Epoch', epoch)
    total_loss = 0
    pred1 = []
    target1 = []
    for cnt, (title, desc, cate1, cate2, cate3, t_len, d_len) in enumerate(dataloader):
        title = title.to(device)
        desc = desc.to(device)
        title = title.to(device)
        desc = desc.to(device)
        cate1 = cate1.to(device)
        optimizer.zero_grad()
        output = model(title, desc, t_len, d_len, mode=1)
        loss = criterion(output, cate1)
        total_loss += loss.item()
        if (cnt + 1) % 100 == 0:
            print('{} / {} Training current loss {:.4}'.format((cnt + 1) * args.batch_size,
                                                             TRAIN_SAMPLES,
                                                             total_loss / (cnt + 1)))
        loss.backward()
        optimizer.step()
        pred1.extend(output.argmax(1).tolist())
        target1.extend(cate1.tolist())
    print('Training Total Loss {:.4}'.format(total_loss / (cnt + 1)))
    cate1_score = f1_score(target1, pred1, average='macro')
    print('Training cate1 f1 score: {:.4}'.format(cate1_score))


def eval_epoch(epoch, model, dataloader, best_score, args):
    model.eval()
    pred1 = []
    target1 = []
    for title, desc, cate1, cate2, cate3, t_len, d_len in dataloader:
        title = title.to(device)
        desc = desc.to(device)
        output = model(title, desc, t_len, d_len, mode=0)
        pred1.extend(output.argmax(1).tolist())
        target1.extend(cate1.tolist())
    cate1_score = f1_score(target1, pred1, average='macro')
    print('Validation cate1 f1 score: {:.4}'.format(cate1_score))
    if cate1_score > best_score:
        print('==> Saving..')
        best_score = cate1_score
        state = {'model': model.state_dict(),
                 'args': args,
                 'best_score': best_score,
                 'epoch': epoch}
        torch.save(state, args.ckpt)
    return  best_score