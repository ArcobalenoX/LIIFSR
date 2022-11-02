import math
import torch
import torch.nn as nn
from models import register
from common import conv, CALayer, PALayer, Upsampler, compute_num_params , Get_gradient_nopadding


class MKRA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.brb_3x3 = nn.Conv2d(n_feats, n_feats//4, kernel_size=3, padding=1)
        self.brb_1x3 = nn.Conv2d(n_feats, n_feats//4, kernel_size=(1, 3), padding=(0, 1))
        self.brb_3x1 = nn.Conv2d(n_feats, n_feats//4, kernel_size=(3, 1), padding=(1, 0))
        self.brb_1x1 = nn.Conv2d(n_feats, n_feats//4, kernel_size=1, padding=0)
        self.act1 = nn.ReLU(True)

        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=(3//2))
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=(3 // 2))
        self.conv3 = nn.Conv2d(n_feats*3, n_feats, 1, padding=(1//2))
        self.ca = CALayer(n_feats, 8)
        self.pa = PALayer(n_feats, 8)

    def forward(self, x):
        iden = x
        x33 = self.brb_3x3(x)
        x13 = self.brb_1x3(x)
        x31 = self.brb_3x1(x)
        x11 = self.brb_1x1(x)
        xm = torch.cat([x33, x13, x31, x11], dim=1)
        y = self.ca(xm)
        y = self.pa(y)
        y = y+iden
        return y

#融合LR梯度图和L0smooth梯度图

#@register('mkdg')
class MKDG(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()
        n_colors = 3

        #define identity branch
        m_identity = []
        m_identity.append(conv(n_colors, n_feats))
        m_identity.append(Upsampler(conv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)
        self.identity_up = conv(n_feats, n_colors)

        # define residual branch
        m_residual = []
        m_residual.append(conv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(MKRA(n_feats))
        m_residual.append(Upsampler(conv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)
        self.residual_up = conv(n_feats, n_colors)

        # define L0 grad branch
        m_smoothgrad = []
        m_smoothgrad.append(conv(n_colors, n_feats))
        m_smoothgrad.append(Upsampler(conv, scale, n_feats))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)
        self.smoothgrad_up = conv(n_feats, n_colors)

        self.get_grad = Get_gradient_nopadding()
        self.grad_up = Upsampler(conv, scale, n_colors)

    def forward(self, x, l0):

        inp = self.identity_up(self.identity(x))
        #print(inp.shape)

        grad = self.get_grad(x)
        grad_up = self.grad_up(grad)
        #print(grad_up.shape)

        res = self.residual(x)
        res_up = self.residual_up(res)
        #print(res.shape)

        l0grad = self.smoothgrad(x)
        l0grad_up = self.smoothgrad_up(l0grad)
        #print(l0grad.shape)

        y = inp + res_up + grad_up + l0grad_up
        return y

@register('mkdg')
def mkdg(n_resblocks=20, n_feats=64, scale=4):
    return MKDG(n_resblocks=n_resblocks, n_feats=n_feats, scale=scale)


@register('mkdg_low')
def mkdg_low(scale=4):
    return MKDG(n_resblocks=10, n_feats=64, scale=scale)

@register('mkdg_mid')
def mkdg_mid(scale=4):
    return MKDG(n_resblocks=15, n_feats=64, scale=scale)

@register('mkdg_high')
def mkdg_high(scale=4):
    return MKDG(n_resblocks=20, n_feats=64, scale=scale)



if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48).cuda()
    l = torch.rand(1, 3, 48, 48).cuda()
    model = MKDG(n_resblocks=20, n_feats=64, scale=4).cuda()
    y = model(x, l)
    print(model)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)

