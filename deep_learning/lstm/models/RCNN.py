import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .conv import Conv


class RCNN(nn.Module):

    def __init__(self, TEXT, LABEL, dropout=0.2, freeze=True):
        super(RCNN, self).__init__()
        hidden_dim = [128, 128, 128]
        self.dropout = dropout
        embedding_dim = TEXT.vocab.vectors.size(1)
        self.embedding = nn.Embedding(len(TEXT.vocab), embedding_dim)
        self.lstm = nn.ModuleList(
            [nn.LSTM(embedding_dim, hidden_dim[i], batch_first=True, bidirectional=True) for i in range(3)])
        self.convs = nn.ModuleList(
            [Conv(embedding_dim, 256, 5, 1, 2, dim=1, numblocks=1) for _ in range(3)])
        self.fcs = nn.ModuleList(
            [nn.Linear(2*hidden_dim[i], len(LABEL[i].vocab)) for i in range(len(LABEL))])
        self._initialize_weights()
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, x, training=True):
        x = self.embedding(x)
        x = [self.lstm[i](x)[0].permute(0, 2, 1).contiguous()
             for i in range(3)]  # [(N, Co, W), ...]*len(Ks)
        x = [self.convs[i](x[i])
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
