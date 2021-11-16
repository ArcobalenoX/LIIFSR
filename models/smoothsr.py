import torch
import torch.nn as nn
from argparse import Namespace

import utils
from models import register
from common import conv, SELayer, Upsampler

class RSEB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = conv(n_feats, n_feats, 3)
        self.act1 = nn.ReLU(True)
        self.conv2 = conv(n_feats, n_feats//2, 3)
        self.conv3 = conv(n_feats+n_feats+n_feats//2, n_feats, 1)
        self.se = SELayer(n_feats, 8)

    def forward(self, x):
        x1 = x
        x2 = self.conv1(x1)
        x2 = self.act1(x2)
        x3 = self.conv2(x2)
        y = torch.cat([x3, x2, x1], dim=1)
        y = self.conv3(y)
        y = self.se(y)+x
        return y

class L0SmoothSR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        kernel_size = 3
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale
        act = False
        #act = "prelu"

        #define identity branch
        m_identity = []
        m_identity.append(Upsampler(conv, scale, args.n_colors, act=act))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(conv(args.n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(RSEB(n_feats))
        m_residual.append(conv(n_feats, args.n_colors, kernel_size))
        m_residual.append(Upsampler(conv, scale, args.n_colors, act=act))
        self.residual = nn.Sequential(*m_residual)

        smooth_iden = []
        smooth_iden.append(conv(args.n_colors, n_feats))
        for _ in range(n_resblocks//2):
            smooth_iden.append(RSEB(n_feats))
        smooth_iden.append(conv(n_feats, args.n_colors, kernel_size))
        smooth_iden.append(Upsampler(conv, scale, args.n_colors, act=act))
        self.smooth = nn.Sequential(*smooth_iden)

        self.out_dim = args.n_colors

    def forward(self, x, l):
        inp = self.identity(x)
        lu = self.smooth(l)
        res = self.residual(x)
        y = res+inp+lu
        return y


@register('L0SmoothSR')
def make_L0SmoothSR(n_resblocks=20, n_feats=64, upsampling=True, scale=2):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.upsampling = upsampling
    args.scale = scale
    args.n_colors = 3
    return L0SmoothSR(args)


if __name__ == '__main__':
    x = torch.rand(1, 3, 128, 128)
    l = torch.rand(1, 3, 128, 128)
    model = make_L0SmoothSR(upsampling=True, scale=4)
    y = model(x,l)
    print(model)
    param_nums = utils.compute_num_params(model,True)
    print(param_nums)
    print(y.shape)

