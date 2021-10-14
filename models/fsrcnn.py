import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

from models import register
from common import compute_num_params

class FSRCNN_net(nn.Module):
    def __init__(self, args):
        input_channels = args.in_ch
        upscale = args.scale
        d = args.d
        s = args.s
        m = args.m

        super().__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.tail_conv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=upscale,
                                            stride=upscale, padding=0)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        out = self.tail_conv(fea)
        return out


@register('fsrcnn')
def make_fsrcnn(input_channels=3, scale=2, d=64, s=12, m=4):
    args = Namespace()
    args.in_ch = input_channels
    args.scale = scale
    args.d = d
    args.s = s
    args.m = m
    return FSRCNN_net(args)


if __name__ == '__main__':
    x = torch.rand(1, 3, 96, 96)
    model = make_fsrcnn(input_channels=3, scale=3)
    y = model(x)
    #print(model)
    print(y.shape)
    print("param_nums:", compute_num_params(model,True))

