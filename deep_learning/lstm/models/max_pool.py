import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class MaxPool(nn.Module):

    def __init__(self, TEXT, LABEL, dropout=0.2, freeze=True):
        super(MaxPool, self).__init__()
        self.dropout = dropout
        embedding_dim = TEXT.vocab.vectors.size(1)
        self.embedding = nn.Embedding(len(TEXT.vocab), embedding_dim)
        self.fcs = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for i in range(len(LABEL))])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(embedding_dim) for i in range(3)])
        self.fcs2 = nn.ModuleList(
            [nn.Linear(embedding_dim, len(LABEL[i].vocab)) for i in range(len(LABEL))])
        self._initialize_weights()
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, x, training=True):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        # (N, len(Ks)*Co)
        x = F.dropout(x, self.dropout,
                      training=training) if self.dropout else x
        x = [self.fcs[i](x) for i in range(3)]  # (N, C)
        x = [F.relu(self.bns[i](x[i])) for i in range(3)]  # (N, C)
        x = [self.fcs2[i](x[i]) for i in range(3)]  # (N, C)
        return x

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
