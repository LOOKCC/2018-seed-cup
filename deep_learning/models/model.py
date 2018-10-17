import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):

    def __init__(self, TEXT, LABEL, dropout=0.2):
        super(TextCNN, self).__init__()
        c = 100
        kernel_sizes = (3, 4, 5)
        embedding_dim = TEXT.vocab.vectors.size(1)
        self.embedding = nn.Embedding(len(TEXT.vocab), embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, c, (k, embedding_dim)) for k in kernel_sizes])
        #self.bns = nn.ModuleList([nn.BatchNorm2d(c) for _ in range(len(kernel_sizes))])
        self.dropout = nn.Dropout(dropout)
        self.fcs = nn.ModuleList([nn.Linear(len(kernel_sizes)*c, len(LABEL[i].vocab)) for i in len(LABEL)])
        self._initialize_weights()
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x).squeeze(3)) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = [fc(x) for fc in self.fcs]  # (N, C)
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
