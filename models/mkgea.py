import math
import time
import torch
import torch.nn as nn
from models import register
from common import conv, CALayer, PALayer, Upsampler, compute_num_params, compute_flops
from gradient import Get_gradient_nopadding
from attention.ExternalAttention import ExternalAttention

"""
多核 梯度 EA注意力
"""

class MKRA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.brb_3x3 = nn.Conv2d(n_feats, n_feats//4, kernel_size=3, padding=1)
        self.brb_1x3 = nn.Conv2d(n_feats, n_feats//4, kernel_size=(1, 3), padding=(0, 1))
        self.brb_3x1 = nn.Conv2d(n_feats, n_feats//4, kernel_size=(3, 1), padding=(1, 0))
        self.brb_1x1 = nn.Conv2d(n_feats, n_feats//4, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(0.01, True)
        self.ca = CALayer(n_feats, n_feats//4)
        self.pa = PALayer(n_feats, n_feats//4)

    def forward(self, x):
        x33 = self.brb_3x3(x)
        x13 = self.brb_1x3(x)
        x31 = self.brb_3x1(x)
        x11 = self.brb_1x1(x)
        xm = torch.cat([x33, x13, x31, x11], dim=1)
        y = self.act(xm)
        y = self.ca(y)
        y = self.pa(y)
        y = y+x
        return y

@register('mkgea')
class MKGEA(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()
        n_colors = 3
        #define identity branch
        m_identity = []
        m_identity.append(conv(n_colors, n_feats))
        m_identity.append(Upsampler(conv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(conv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(MKRA(n_feats))
        m_residual.append(Upsampler(conv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        # define grad branch
        m_gradbranch = []
        m_gradbranch.append(conv(n_colors, n_feats))
        m_gradbranch.append(Upsampler(conv, scale, n_feats))
        self.gradbranch = nn.Sequential(*m_gradbranch)

        self.fusion = conv(n_feats*3, n_colors)
        self.ea = ExternalAttention(d_model=n_feats, S=8)

    def forward(self, x):
        inp = self.identity(x)

        gard = Get_gradient_nopadding()(x)
        grad = self.gradbranch(gard)
        res = self.residual(x)

        res = res.permute(0, 2, 3, 1)
        res = self.ea(res)
        res = res.permute(0, 3, 1, 2)

        y = torch.cat([inp, res, grad], dim=1)
        y = self.fusion(y)
        return y



if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48).cuda()
    model = MKGEA(n_resblocks=10, n_feats=32, scale=4).cuda()
    t = time.time()

    y = model(x)
    #print(model)
    print("time ",time.time()-t)
    param_nums = compute_num_params(model, True)
    print(param_nums)
    print(y.shape)

    #compute_flops(model,x)

