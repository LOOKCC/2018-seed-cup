import torch
import torch.nn as nn
import torch.nn.init as init

class GLU(nn.Module):
    def __init__(self, embedding_dim, k):
        super(GLU, self).__init__()
        self.conv = nn.Conv2d(1, 2 * embedding_dim, (k, embedding_dim), padding=(k//2, 0))
        init.xavier_normal_(self.conv.weight.data)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()

    def forward(self, X):
        X = X.unsqueeze(1)  # (B, 1, L, D)
        y = self.conv(X)  # (B, 2D, L, 1)
        y = y.squeeze(-1).transpose(1, 2)  # (B, L, 2D)
        y = y[:, :, :int(y.size(-1) / 2)] * torch.sigmoid(y[:, :, int(y.size(-1) / 2):])  # (B, L, D)
        return y

