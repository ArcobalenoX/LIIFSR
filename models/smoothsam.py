import torch
import torch.nn as nn
from argparse import Namespace

import utils
from models import register
from common import conv, SELayer, Upsampler, SAM, PALayer


class RSEB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = conv(n_feats, n_feats, 3)
        self.act1 = nn.ReLU(True)
        self.conv2 = conv(n_feats, n_feats // 2, 3)
        self.conv3 = conv(n_feats + n_feats + n_feats // 2, n_feats, 1)
        self.att = PALayer(n_feats, 8)

    def forward(self, x):
        x1 = x
        x2 = self.conv1(x1)
        x2 = self.act1(x2)
        x3 = self.conv2(x2)
        y = torch.cat([x3, x2, x1], dim=1)
        y = self.conv3(y)
        y = self.att(y)+x
        return y

class L0SmoothSR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        kernel_size = 3
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale

        #define identity branch
        m_identity = []
        m_identity.append(Upsampler(conv, scale, args.n_colors))
        self.identity = nn.Sequential(*m_identity)


        # define residual branch
        m_residual = []
        m_residual.append(conv(args.n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(RSEB(n_feats))
        m_residual.append(conv(n_feats, args.n_colors, kernel_size))
        m_residual.append(Upsampler(conv, scale, args.n_colors))
        self.residual = nn.Sequential(*m_residual)

        m_smoothgrad = []
        m_smoothgrad.append(Upsampler(conv, scale, args.n_colors))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.sam = SAM(args.n_colors, kernel_size, True)

        self.ch = conv(1, 3, kernel_size)
        self.fusion = conv(9, 3, kernel_size)


    def forward(self, x, l):
        inp = self.identity(x)
        lu = self.smoothgrad(l)
        res = self.residual(x)
        sam, attgrad = self.sam(inp, lu)
        y = torch.cat([inp, sam, res], dim=1)
        y = self.fusion(y)
        return y


@register('L0Smoothsam')
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
    model = make_L0SmoothSR(scale=2)
    y = model(x,l)
    print(model)
    param_nums = utils.compute_num_params(model,True)
    print(param_nums)
    print(y.shape)

