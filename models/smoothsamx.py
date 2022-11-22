import torch
import torch.nn as nn
from models import register
from common import compute_num_params, conv, Upsampler, PALayer
#小论文使用
class RSPA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = conv(n_feats, n_feats, 3)
        self.act1 = nn.ReLU(True)
        self.conv2 = conv(n_feats, n_feats, 3)
        self.conv3 = conv(n_feats*3, n_feats, 1)
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


@register('L0Smoothsamx')
class L0SmoothSR(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()
        kernel_size = 3
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
            m_residual.append(RSPA(n_feats))
        m_residual.append(Upsampler(conv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        # define grad branch
        m_smoothgrad = []
        m_smoothgrad.append(conv(n_colors, n_feats))
        m_smoothgrad.append(Upsampler(conv, scale, n_feats))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.fusion = conv(n_feats*3, 3, kernel_size)


    def forward(self, x, l):
        inp = self.identity(x)
        lu = self.smoothgrad(l)
        res = self.residual(x)
        y = torch.cat([inp, lu, res], dim=1)
        y = self.fusion(y)
        return y


@register('L0Smoothlow')
def L0Smooth_low(scale=4):
    return L0SmoothSR(n_resblocks=10, n_feats=32, scale=scale)

@register('L0Smoothmid')
def L0Smooth_mid(scale=4):
    return L0SmoothSR(n_resblocks=20, n_feats=32, scale=scale)

@register('L0Smoothhigh')
def L0Smooth_high(scale=4):
    return L0SmoothSR(n_resblocks=20, n_feats=64, scale=scale)


if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48)
    l = torch.rand(1, 3, 48, 48)
    model = L0SmoothSR(20, 64, scale=4)
    y = model(x, l)
    print(model)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)

