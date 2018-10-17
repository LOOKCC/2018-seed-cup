import torch
import torch.nn as nn
import torch.nn.init as init


class Conv(nn.Module):
    """
    A template of conv layers

    Args:
        dim (int, optional): 1d or 2d or 3d ? (default: 2)
        numblocks (int, optional): Number of layers to concat (default: 1)
        transpose (bool, optional): Deconv(transposed conv) if True (default: False)
    """
    def __init__(self, in_chans, out_chans, kernel, stride=1, padding=0, dilate=1,
                 dim=2, numblocks=1, transpose=False, relu=True, bn=True, gn=False, coord=False):
        super().__init__()
        conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d,
                nn.ConvTranspose2d, nn.ConvTranspose3d][dim - 1 + transpose*3]
        batchnorm = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        self.coord = coord and dim==2
        if self.coord: in_chans += 2
        layers = []
        for block_id in range(numblocks):
            layers.append(conv(in_chans, out_chans, kernel, stride, padding, dilate))
            if bn: layers.append(batchnorm[dim-1](out_chans))
            if gn: layers.append(nn.GroupNorm(gn, out_chans))
            if relu: layers.append(nn.ReLU(inplace=True))
            in_chans = out_chans
        self.cnns = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        if self.coord:
            x = self._add_coord2d(x)
        x = self.cnns(x)
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

    def _add_coord2d(self, x):
        n, c, h, w = x.size()
        x_channel = torch.arange(w).repeat(n, 1, h, 1)
        y_channel = torch.arange(h).repeat(n, 1, w, 1).transpose(2, 3)
        x_channel = (x_channel.float()/(w-1)).to(x.device)
        y_channel = (y_channel.float()/(h-1)).to(x.device)
        return torch.cat((x, x_channel, y_channel), dim=1)
