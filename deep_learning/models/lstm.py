import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self, hidden_dim, TEXT, LABEL, dropout=0.2, freeze=True):
        super(LSTM, self).__init__()
        self.dropout = dropout
        embedding_dim = TEXT.vocab.vectors.size(1)
        self.embedding = nn.Embedding(len(TEXT.vocab), embedding_dim)
        self.lstm = nn.ModuleList(
            [nn.LSTM(embedding_dim, hidden_dim[i], batch_first=True, bidirectional=True) for i in range(3)])
        self.lns = nn.ModuleList(
            [nn.LayerNorm(2*hidden_dim[i]) for i in range(3)])
        # self.fcs = nn.ModuleList(
        #     [nn.Linear(2*hidden_dim[i], 2*hidden_dim[i]) for i in range(len(LABEL))])
        # self.bns = nn.ModuleList([nn.BatchNorm1d(2*hidden_dim[i]) for i in range(3)])
        self.fcs = nn.ModuleList(
            [nn.Linear(2*hidden_dim[i], len(LABEL[i].vocab)) for i in range(len(LABEL))])
        self._initialize_weights()
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, x, training=True):
        x = self.embedding(x)
        x = [self.lns[i](self.lstm[i](x)[0]).permute(0, 2, 1).contiguous()
             for i in range(3)]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(x[i], x[i].size(2)).squeeze(2)
             for i in range(3)]  # [(N, Co), ...]*len(Ks)

        x = [F.dropout(x[i], self.dropout, training=training)
             for i in range(3)] if self.dropout else x  # (N, len(Ks)*Co)
        x = [self.fcs[i](x[i]) for i in range(3)]  # (N, C)
        # x = [F.relu(self.bns[i](x[i])) for i in range(3)]  # (N, C)
        # x = [self.fcs2[i](x[i]) for i in range(3)]  # (N, C)
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
