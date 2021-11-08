import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register

from common import default_conv, MeanShift, Upsampler, PALayer
from common import compute_num_params


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        self.conv = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                  act,
                                  conv(n_feat, n_feat, kernel_size, bias=bias)
                                  )
        self.res_scale = res_scale
        self.att = PALayer(n_feat, 8)

    def forward(self, x):
        y = self.conv(x)
        y = self.att(y)
        y = y * self.res_scale + x
        return y


class EDSRPA(nn.Module):
    def __init__(self, args, conv=default_conv):
        super().__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)


        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.upsampling:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)
        else:
            self.out_dim = n_feats

    def forward(self, x):

        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.upsampling:
            x = self.tail(res)
        else:
            x = res

        return x


@register('edsrpa')
def make_edsrpa(n_resblocks=32, n_feats=256, res_scale=0.1,
              scale=2, upsampling=True, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.scale = scale
    args.upsampling = upsampling
    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSRPA(args)


if __name__ == '__main__':
    x = torch.rand(1, 3, 128, 128)
    model = make_edsrpa(upsampling=False, scale=2, n_resblocks=16, n_feats=64)
    y = model(x)
    print(model)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)

