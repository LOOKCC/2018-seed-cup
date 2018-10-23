import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .lstm import LSTM


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        hidden_dims = [[128, 128, 128]]
        self.models = nn.ModuleList([LSTM(hidden_dims[i], *args, **kwargs)
                                     for i in range(len(hidden_dims))])

    def forward(self, batch, training=True):
        x = [torch.cat((batch.title_chars, ), dim=1)]
        # x = [batch.disc_words]
        x = [self.models[i](x[i]) for i in range(len(x))]
        result = zip(*x)
        result = [sum(prob) for prob in result]
        return x, result
