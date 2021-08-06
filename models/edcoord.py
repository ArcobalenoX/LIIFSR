import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

import utils
from models import register


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


def default_conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(self, n_feats, act=nn.ReLU(True), res_scale=1):

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


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, act=None, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDCOORD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        kernel_size = 3
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        scale = args.scale
        act = nn.ReLU(True)

        # define head module
        m_head = []
        m_head.append(default_conv(args.n_colors, n_feats))
        m_head.append(CoordAtt(n_feats, n_feats, 16))
        self.head = nn.Sequential(*m_head)

        # define body module
        m_body = []
        for _ in range(n_resblocks):
            m_body.append(ResBlock(n_feats))
            m_body.append(CoordAtt(n_feats, n_feats, 32))
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
                    raise RuntimeError(f'While copying the parameter named "{name}", \
                                        whose dimensions in the model are "{own_state[name].size()}" \
                                        and whose dimensions in the checkpoint are "{param.size()}".')
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')

@register('edcoord')
def make_edcoord(n_resblocks=16, n_feats=64, upsampling=False, scale=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.upsampling = upsampling
    args.scale = scale
    args.n_colors = 3
    return EDCOORD(args)


if __name__ == '__main__':
    x = torch.rand(1, 3, 128, 128)
    #x = Image.open('../09_1.jpg')
    #x = ToTensor()(x).unsqueeze(0)
    model = make_edcoord(upsampling=True, scale=2)
    y = model(x)
    print(model)
    param_nums = utils.compute_num_params(model)
    print(param_nums)
    print(y.shape)
    #y = ToPILImage()(y.squeeze()).save('09_1_edmk.jpg')

