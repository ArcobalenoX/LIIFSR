import math
import torch
import torch.nn as nn
from models import register
from common import conv, Upsampler, CoordAtt, compute_num_params


class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.brb_3x3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.brb_1x3 = nn.Conv2d(n_feats, n_feats, kernel_size=(1, 3), padding=(0, 1))
        self.brb_3x1 = nn.Conv2d(n_feats, n_feats, kernel_size=(3, 1), padding=(1, 0))
        self.brb_1x1 = nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=(3 // 2))
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=(3 // 2))
        self.act2 = nn.ReLU(True)
        self.att = CoordAtt(n_feats, n_feats, 4)

    def forward(self, x):
        x1 = x
        x33 = self.brb_3x3(x)
        x13 = self.brb_1x3(x)
        x31 = self.brb_3x1(x)
        x11 = self.brb_1x1(x)
        xm = (x33 + x13 + x31 + x11)
        x2 = self.act1(xm)
        x3 = self.conv2(x2)
        x4 = self.att(x2)
        y = x1 + x3 + x4
        y = self.act2(y)
        return y

@register('drsenmkca')
class DRSENMKCA(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()

        kernel_size = 3
        n_colors = 3
        act = nn.ReLU(True)

        # define identity branch
        m_identity = []
        m_identity.append(Upsampler(conv, scale, n_colors, act=False))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(conv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(ResBlock(n_feats))
        m_residual.append(Upsampler(conv, scale, n_feats, act=False))
        m_residual.append(conv(n_feats, n_colors, kernel_size))
        self.residual = nn.Sequential(*m_residual)
        self.out_dim = n_colors

    def forward(self, x):
        inp = self.identity(x)
        res = self.residual(x)
        y = res + inp
        return y



if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48)
    model = DRSENMKCA(scale=4)
    y = model(x)
    print(model)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)
