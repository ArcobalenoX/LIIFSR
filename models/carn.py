import torch
import torch.nn as nn
import common

from models import register
from common import compute_num_params


class Block(nn.Module):
    def __init__(self, nf, group=1):
        super().__init__()

        self.b1 = common.EResidualBlock(nf, nf, group=group)
        self.c1 = common.BasicBlock(nf * 2, nf, 1, 1, 0)
        self.c2 = common.BasicBlock(nf * 3, nf, 1, 1, 0)
        self.c3 = common.BasicBlock(nf * 4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


@register('carn')
class CARN_M(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, scale=4, multi_scale=False, group=4):
        super().__init__()
        self.scale = scale
        # rgb_range = 1
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        # self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.b1 = Block(nf, group=group)
        self.b2 = Block(nf, group=group)
        self.b3 = Block(nf, group=group)
        self.c1 = common.BasicBlock(nf * 2, nf, 1, 1, 0)
        self.c2 = common.BasicBlock(nf * 3, nf, 1, 1, 0)
        self.c3 = common.BasicBlock(nf * 4, nf, 1, 1, 0)

        self.upsample = common.UpsampleBlock(nf, scale=scale,
                                                multi_scale=multi_scale,
                                                group=group)
        self.exit = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        # out = self.add_mean(out)

        return out

if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48)
    model = CARN_M(scale=2)
    y = model(x)
    print(model)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)