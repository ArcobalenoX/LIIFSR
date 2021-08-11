import math
import torch
import torch.nn as nn
from argparse import Namespace

import utils
from models import register

from CoordAtt import CoordAtt
from common import default_conv, Upsampler


class RMKCA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.brb_3x3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.brb_1x3 = nn.Conv2d(n_feats, n_feats, kernel_size=(1, 3), padding=(0, 1))
        self.brb_3x1 = nn.Conv2d(n_feats, n_feats, kernel_size=(3, 1), padding=(1, 0))
        self.brb_1x1 = nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0)
        # self.norm = nn.InstanceNorm2d(n_feats)
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=(3//2))
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=(3//2))
        self.conv3 = nn.Conv2d(n_feats*3, n_feats, 1, padding=(1//2))
        self.ca = CoordAtt(n_feats, n_feats, 8)

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
        y = self.ca(y)+x

        return y

class MKCA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        kernel_size = 3
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale
        act = nn.ReLU(True)

        #define identity branch
        m_identity = []
        m_identity.append(default_conv(args.n_colors, n_feats, kernel_size))
        m_identity.append(Upsampler(default_conv, scale, n_feats, act=False))
        m_identity.append(default_conv(n_feats, args.n_colors, kernel_size))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(default_conv(args.n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(RMKCA(n_feats))
        m_residual.append(Upsampler(default_conv, scale, n_feats, act=False))
        m_residual.append(default_conv(n_feats, args.n_colors, kernel_size))
        self.residual = nn.Sequential(*m_residual)

        self.out_dim = args.n_colors

    def forward(self, x):
        inp = self.identity(x)
        res = self.residual(x)
        y = res + inp
        return y


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(f'While copying the parameter named "{name}", \
                                        whose dimensions in the model are "{own_state[name].size()}" \
                                        and whose dimensions in the checkpoint are "{param.size()}".')
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')

@register('mkca')
def make_mkca(n_resblocks=16, n_feats=64, upsampling=False, scale=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.upsampling = upsampling
    args.scale = scale
    args.n_colors = 3
    return MKCA(args)


if __name__ == '__main__':
    x = torch.rand(1, 3, 128, 128)
    model = make_mkca(upsampling=True, scale=2)
    y = model(x)
    print(model)
    param_nums = utils.compute_num_params(model)
    print(param_nums)
    print(y.shape)


