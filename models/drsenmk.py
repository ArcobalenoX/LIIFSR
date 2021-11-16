import math
import torch
import torch.nn as nn
from models import register
from common import conv, SELayer, Upsampler, compute_num_params


class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.brb_3x3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.brb_1x3 = nn.Conv2d(n_feats, n_feats, kernel_size=(1, 3), padding=(0, 1))
        self.brb_3x1 = nn.Conv2d(n_feats, n_feats, kernel_size=(3, 1), padding=(1, 0))
        self.brb_1x1 = nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=(3//2))
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=(3//2))
        self.conv3 = nn.Conv2d(n_feats*3, n_feats, 1, padding=(1//2))
        self.att = SELayer(n_feats, 8)

    def forward(self, x):
        x1 = x
        x33 = self.brb_3x3(x)
        x13 = self.brb_1x3(x)
        x31 = self.brb_3x1(x)
        x11 = self.brb_1x1(x)
        xm = (x33 + x13 + x31 + x11)
        x2 = self.act1(xm)
        x3 = self.conv2(x2)
        #x4 = self.conv1(x3)
        y = torch.cat([x3, x2, x1], dim=1)
        y = self.conv3(y)
        y = self.att(y)+x

        return y

@register('drsenmk')
class DRSENMK(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()

        kernel_size = 3
        n_colors = 3
        act = nn.ReLU(True)

        #define head module
        m_head = []
        m_head.append(conv(n_colors, n_feats))
        m_head.append(Upsampler(conv, scale, n_feats, act=False))
        m_head.append(conv(n_feats, n_colors, kernel_size))
        self.head = nn.Sequential(*m_head)


        # define body module
        m_body = []
        m_body.append(conv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_body.append(ResBlock(n_feats))
        self.body = nn.Sequential(*m_body)

        # define tail module
        m_tail = []
        m_tail.append(Upsampler(conv, scale, n_feats, act=False))
        m_tail.append(conv(n_feats, n_colors, kernel_size))
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        inp = self.head(x)
        res = self.body(x)
        res = self.tail(res)
        x = res+inp
        return x





if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48)
    model = DRSENMK(scale=4)
    y = model(x)
    print(model)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)

