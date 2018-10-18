import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .conv import Conv


class TextCNN(nn.Module):

    def __init__(self, TEXT, LABEL, dropout=0.2, freeze=True):
        super(TextCNN, self).__init__()
        c = 512
        embedding_dim = TEXT.vocab.vectors.size(1)
        self.embedding = nn.Embedding(len(TEXT.vocab), embedding_dim)
        self.convs = Conv(embedding_dim, c, 3, 1, 1, dim=1, numblocks=3)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fcs = nn.ModuleList([nn.Linear(len(kernel_sizes)*c, len(LABEL[i].vocab)) for i in range(len(LABEL))])
        self._initialize_weights()
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)
        if freeze:
            self.embedding.weight.requires_grad=False

    def forward(self, x):
        x = self.embedding(x)
        x = self.convs(x)  # [(N, Co, W), ...]*len(Ks)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) # [(N, Co), ...]*len(Ks)

        x = x if self.dropout is None else self.dropout(x)  # (N, len(Ks)*Co)
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
