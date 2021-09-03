import math
import torch
import torch.nn as nn
from argparse import Namespace

import utils
from models import register
from OutlookAttention import OutlookAttention
from CoordAtt import CoordAtt
from common import default_conv, SELayer, Upsampler


class ResBlock(nn.Module):
    def __init__(self, n_feats, IN=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(default_conv(n_feats, n_feats))
            if IN:
                m.append(nn.InstanceNorm2d(n_feats))
            m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res



class EDCA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        kernel_size = 3
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale[0]
        act = nn.ReLU(True)

        # define head module
        m_head = []
        m_head.append(default_conv(args.n_colors, n_feats))
        m_head.append(CoordAtt(n_feats, n_feats, 16))
        self.head = nn.Sequential(*m_head)

        # define body module
        m_body = []
        for _ in range(n_resblocks):
            m_body.append(ResBlock(n_feats, IN=True, res_scale=args.res_scale) )
            m_body.append(CoordAtt(n_feats, n_feats, 32))
            m_body.append(OutlookAttention(n_feats))
        self.body = nn.Sequential(*m_body)

        if args.upsampling:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = []
            m_tail.append(Upsampler(default_conv, scale, n_feats, act=False))
            m_tail.append(default_conv(n_feats, args.n_colors, kernel_size))
            self.tail = nn.Sequential(*m_tail)
        else:
            self.out_dim = n_feats

    def forward(self, x):

        inp = x

        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.upsampling:
            x = self.tail(res)
        else:
            x = res

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    #if name.find('tail') == -1:
                    raise RuntimeError(f'While copying the parameter named "{name}", \
                                        whose dimensions in the model are "{own_state[name].size()}" \
                                        and whose dimensions in the checkpoint are "{param.size()}".')
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')

@register('edca')
def make_edca(n_resblocks=16, n_feats=64, res_scale=1, scale=1, upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.scale = [scale]
    args.upsampling = upsampling
    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDCA(args)


if __name__ == '__main__':
    x = torch.rand(1, 3, 128, 128)
    model = make_edca(upsampling=True, scale=2)
    y = model(x)
    print(model)
    param_nums = utils.compute_num_params(model)
    print(param_nums)
    print(y.shape)

