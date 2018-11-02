import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os

from sklearn.metrics import f1_score

from .glu import GLU
from utils.timer import timer

CATE1_CNT = 20
CATE2_CNT = 135
CATE3_CNT = 265

TRAIN_SAMPLES = 911256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Cate1Classifier(nn.Module):
    def __init__(self, input_size, args, embedding_weight=None):
        super(Cate1Classifier, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(input_size, args.embedding_dim, padding_idx=0)
        self.glu1 = GLU(args.embedding_dim, 3)
        self.glu2 = GLU(args.embedding_dim, 5)
        self.fc = nn.Linear(args.embedding_dim*4, args.h_d)
        self.clf = nn.Linear(args.h_d, CATE1_CNT)
        self.bn1 = nn.BatchNorm1d(args.embedding_dim*4)
        self.bn2 = nn.BatchNorm1d(args.h_d)
        self._initialize_weights()
        if embedding_weight is not None:
            self.embedding.weight.data.copy_(embedding_weight)

    def forward(self, title, desc):
        title_vec = self.embedding(title)
        title_vec1 = self.glu1(title_vec)
        title_vec1, _ = title_vec1.max(1)
        title_vec2 = self.glu2(title_vec)
        title_vec2, _ = title_vec2.max(1)
        desc_vec = self.embedding(desc)
        desc_vec1 = self.glu1(desc_vec)
        desc_vec1, _ = desc_vec1.max(1)
        desc_vec2 = self.glu2(desc_vec)
        desc_vec2, _ = desc_vec2.max(1)
        glu_vec = torch.cat([title_vec1, desc_vec1, title_vec2, desc_vec2], 1)
        glu_vec = self.bn1(glu_vec)
        h = self.bn2(self.fc(glu_vec))
        h = F.relu(h)
        output = self.clf(h)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

@timer
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
        output = model(title, desc)
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


def eval_epoch(epoch, model, dataloader, best_score, optimizer, args):
    model.eval()
    pred1 = []
    target1 = []
    for title, desc, cate1, cate2, cate3, t_len, d_len in dataloader:
        title = title.to(device)
        desc = desc.to(device)
        output = model(title, desc)
        pred1.extend(output.argmax(1).tolist())
        target1.extend(cate1.tolist())
    cate1_score = f1_score(target1, pred1, average='macro')
    print('Validation cate1 f1 score: {:.4}'.format(cate1_score))
    if cate1_score > best_score:
        print('==> Saving..')
        best_score = cate1_score
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'args': model.args,
                 'best_score': best_score,
                 'epoch': epoch}
        torch.save(state, os.path.join('checkpoint', args.ckpt))
    return  best_score