import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .lstm import LSTM
from .gru import GRU
from .res_lstm import ResLSTM
from .max_pool import MaxPool
from .textcnn import TextCNN
from .rescnn import ResCNN
from .cnn import CNN
from .rcnn import RCNN


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        kernel_size = [5, 4, 3, 5, 4, 3]
        hidden_dims = [[128, 128, 128]]  # , [128, 128, 128]]
        self.models = nn.ModuleList([CNN(kernel_size[i], *args, **kwargs)
                                     for i in range(len(kernel_size))])
        # if using lstm or gru         GRU
        # self.models = nn.ModuleList([LSTM(hidden_dims[i], *args, **kwargs)
        #                              for i in range(len(hidden_dims))])

    def forward(self, batch, training=True):
        x = ([torch.cat((batch.title_words, batch.disc_words), dim=1)]*3 +
             [torch.cat((batch.title_chars, batch.disc_chars), dim=1)]*3)
        # if using lstm or gru
        # x = [torch.cat((batch.title_words, batch.disc_words), dim=1)]
        y = []
        for i in range(len(x)):
            output = self.models[i](x[i])
            if isinstance(output[0], list):
                y.extend(output)
            else:
                y.append(output)
        del x
        del output
        result = zip(*y)
        result = [sum(prob) for prob in result]
        return y, result
