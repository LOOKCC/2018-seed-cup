import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):
    
    def __init__(self, class_num, embedding_dim, dropout=0.2):
        super(CNN_Text, self).__init__()
        c = 256
        kernel_sizes = (3, 4, 5)
        self.convs = nn.ModuleList([nn.Conv2d(1, c, (k, embedding_dim)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes)*c, class_num)

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc(x)  # (N, C)
        return logit