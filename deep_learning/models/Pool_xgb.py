import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .conv import Conv


class Pool(nn.Module):

    def __init__(self, TEXT, LABEL, dropout=0.2, freeze=True):
        super(Pool, self).__init__()
        cnn_out = 32
        rnn_out = 64
        self.dropout = dropout
        embedding_dim = TEXT.vocab.vectors.size(1)
        self.embedding = nn.Embedding(len(TEXT.vocab), embedding_dim)

        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, x, training=True):
        x = self.embedding(x)
        # n cout w
        x = torch.cat((self.swem_max(x), self.swem_avg(x)), 1)
        return x

    def swem_max(self, input):
        try:
            output, _ = input.max(1)
        except:
            print(input)
            raise
        return output

    def swem_avg(self, input):
        output = input.mean(1)
        return output