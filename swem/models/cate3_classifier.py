import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

from .represent_layer import SwemCat
from utils.timer import timer

WORDS_CNT = 34835
CHARS_CNT = 3939

CATE1_CNT = 10
CATE2_CNT = 64
CATE3_CNT = 125

TRAIN_SAMPLES = 140562

embedding_dim = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Cate3Classifier(nn.Module):
    def __init__(self, args, word2vec=None, mask2=None):
        super(Cate3Classifier, self).__init__()
        self.swem_layer = SwemCat(word2vec)
        self.fc = nn.Linear(embedding_dim*4, args.h_d)
        self.clf = nn.Linear(args.h_d, CATE3_CNT)
        self.bn = nn.BatchNorm1d(args.h_d)
        self.mask2 = mask2

    def forward(self, title, desc, t_len, d_len, cate2, mode=1):
        swem_vec = self.swem_layer(title, desc, t_len, d_len, mode)
        h = self.bn(self.fc(swem_vec))
        h = F.relu(h)
        output = self.clf(h)
        for i in range(title.size(0)):
            output[i, self.mask2[cate2[i]]] = -100
        return output

@timer
def train_epoch(epoch, model, dataloader, criterion, optimizer, args):
    model.train()
    print('Epoch', epoch)
    total_loss = 0
    pred3 = []
    target3 = []
    for cnt, (title, desc, cate1, cate2, cate3, t_len, d_len) in enumerate(dataloader):
        title = title.to(device)
        desc = desc.to(device)
        title = title.to(device)
        desc = desc.to(device)
        cate2 = cate2.to(device)
        cate3 = cate3.to(device)
        optimizer.zero_grad()
        output = model(title, desc, t_len, d_len, cate2, mode=1)
        loss = criterion(output, cate3)
        total_loss += loss.item()
        if (cnt + 1) % 100 == 0:
            print('{} / {} Training current loss {:.4}'.format((cnt + 1) * args.batch_size,
                                                             TRAIN_SAMPLES,
                                                             total_loss / (cnt + 1)))
        loss.backward()
        optimizer.step()
        pred3.extend(output.argmax(1).tolist())
        target3.extend(cate3.tolist())
    print('Training Total Loss {:.4}'.format(total_loss / (cnt + 1)))
    cate3_score = f1_score(target3, pred3, average='macro')
    print('Training cate3 f1 score: {:.4}'.format(cate3_score))

def eval_epoch(epoch, model, dataloader, best_score, args):
    model.eval()
    pred3 = []
    target3 = []
    for title, desc, cate1, cate2, cate3, t_len, d_len in dataloader:
        title = title.to(device)
        desc = desc.to(device)
        cate2 = cate2.to(device)
        output = model(title, desc, t_len, d_len, cate2, mode=0)
        pred3.extend(output.argmax(1).tolist())
        target3.extend(cate3.tolist())
    cate3_score = f1_score(target3, pred3, average='macro')
    print('Validation cate3 f1 score: {:.4}'.format(cate3_score))
    if cate3_score > best_score:
        print('==> Saving..')
        best_score = cate3_score
        state = {'model': model.state_dict(),
                 'args': args,
                 'best_score': best_score,
                 'epoch': epoch}
        torch.save(state, args.ckpt)
    return  best_score