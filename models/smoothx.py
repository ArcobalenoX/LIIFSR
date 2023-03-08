import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from thop import profile
from models import register
from common import compute_num_params, normalconv, Upsampler, PALayer, SELayer
from pan import SCPA, PAConv
from attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention, SequentialPolarizedSelfAttention
from vapsr import VAB
from attention.A2Atttention import DoubleAttention

# 小论文使用
class RSPA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = normalconv(n_feats, n_feats, 3)
        self.act1 = nn.ReLU(True)
        self.conv2 = normalconv(n_feats, n_feats, 3)
        self.conv3 = normalconv(n_feats * 3, n_feats, 1)
        self.att = PALayer(n_feats, 8)

    def forward(self, x):
        x1 = x
        x2 = self.conv1(x1)
        x2 = self.act1(x2)
        x3 = self.conv2(x2)
        y = torch.cat([x3, x2, x1], dim=1)
        y = self.conv3(y)
        y = self.att(y) + x
        return y


class RS(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = normalconv(n_feats, n_feats, 3)
        self.act1 = nn.LeakyReLU(0.01, True)
        self.conv2 = normalconv(n_feats, n_feats, 3)
        self.conv3 = normalconv(n_feats * 3, n_feats, 1)

    def forward(self, x):
        x1 = x
        x2 = self.conv1(x1)
        x2 = self.act1(x2)
        x3 = self.conv2(x2)
        y = torch.cat([x3, x2, x1], dim=1)
        y = self.conv3(y)
        y = y + x
        return y


@register('L0Smoothsamx')
class L0Smoothsamx(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()
        kernel_size = 3
        n_colors = 3
        # define identity branch
        m_identity = []
        m_identity.append(normalconv(n_colors, n_feats))
        m_identity.append(Upsampler(normalconv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(normalconv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(RSPA(n_feats))
        m_residual.append(Upsampler(normalconv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        # define grad branch
        m_smoothgrad = []
        m_smoothgrad.append(normalconv(n_colors, n_feats))
        m_smoothgrad.append(Upsampler(normalconv, scale, n_feats))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.fusion = normalconv(n_feats * 3, 3, kernel_size)

    def forward(self, x, l):
        inp = self.identity(x)
        lu = self.smoothgrad(l)
        res = self.residual(x)
        y = torch.cat([inp, lu, res], dim=1)
        y = self.fusion(y)
        return y


class MKRB(nn.Module):
    def __init__(self, n_feats, reduction=2):
        super().__init__()
        self.brb_3x3 = nn.Conv2d(n_feats, n_feats // reduction, kernel_size=(3, 3), padding=1)
        self.brb_1x3 = nn.Conv2d(n_feats, n_feats // reduction, kernel_size=(1, 3), padding=(0, 1))
        self.brb_3x1 = nn.Conv2d(n_feats, n_feats // reduction, kernel_size=(3, 1), padding=(1, 0))
        self.brb_1x1 = nn.Conv2d(n_feats, n_feats // reduction, kernel_size=(1, 1), padding=0)
        self.act = nn.LeakyReLU(0.01, True)
        self.conv2 = normalconv(n_feats // reduction * 4, n_feats, 3)
        self.conv3 = normalconv(n_feats * 2, n_feats, 1)

    def forward(self, x):
        x33 = self.brb_3x3(x)
        x13 = self.brb_1x3(x)
        x31 = self.brb_3x1(x)
        x11 = self.brb_1x1(x)
        xm = torch.cat([x33, x13, x31, x11], dim=1)
        xm = self.act(xm)
        xm = self.conv2(xm)
        y = torch.cat([xm, x], dim=1)
        y = self.conv3(y)
        y = y + x
        return y


class MKRBPA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.mkrb = MKRB(n_feats)
        self.pa = PALayer(n_feats, n_feats//4)

    def forward(self, x):
        z = self.mkrb(x)
        z = self.pa(z)
        y = x + z
        return y


class ParallePSA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.rspa = RSPA(n_feats)
        self.psa = ParallelPolarizedSelfAttention(n_feats)

    def forward(self, x):
        z = self.rspa(x)
        m = self.psa(x)
        y = x + z + m
        return y


class SequentialPSA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.rspa = RSPA(n_feats)
        self.psa = SequentialPolarizedSelfAttention(n_feats)

    def forward(self, x):
        z = self.rspa(x)
        m = self.psa(x)
        y = x + z + m
        return y


class RSVAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.rs = RS(n_feats)
        self.rf = VAB(n_feats, n_feats)

    def forward(self, x):
        z = self.rs(x)
        z = self.rf(z)
        y = x + z
        return y

class RSDA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.rs = RS(n_feats)
        self.rf = DoubleAttention(n_feats, n_feats, n_feats)

    def forward(self, x):
        z = self.rs(x)
        z = self.rf(z)
        y = x + z
        return y

@register('L0smooth')
class L0smooth(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2, block='none'):
        super().__init__()

        # define identity branch
        m_identity = []
        m_identity.append(normalconv(3, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(normalconv(3, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(eval(block)(n_feats))
        self.residual = nn.Sequential(*m_residual)

        # define grad branch
        m_smoothgrad = []
        m_smoothgrad.append(normalconv(3, n_feats))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.fusion = normalconv(n_feats * 3, n_feats)
        self.up = Upsampler(normalconv, scale, n_feats)
        self.rebuild = normalconv(n_feats, 3)


    def forward(self, x, l):
        inp = self.identity(x)


        res = self.residual(x)

        lu = self.smoothgrad(l)
        y = torch.cat([inp, res, lu], dim=1)

        y = self.fusion(y)
        y = self.up(y)
        y = self.rebuild(y)

        return y

@register('L0smoothgpsa')
class L0smoothgpsa(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2, block='none'):
        super().__init__()

        # define identity branch
        m_identity = []
        m_identity.append(normalconv(3, n_feats))
        self.identity = nn.Sequential(*m_identity)

        self.psa = SequentialPolarizedSelfAttention(n_feats)

        # define residual branch
        m_residual = []
        m_residual.append(normalconv(3, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(RSPA(n_feats))
        self.residual = nn.Sequential(*m_residual)

        # define grad branch
        m_smoothgrad = []
        m_smoothgrad.append(normalconv(3, n_feats))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.fusion = normalconv(n_feats * 4, n_feats)
        self.up = Upsampler(normalconv, scale, n_feats)
        self.rebuild = normalconv(n_feats, 3)

    def forward(self, x, l):
        inp = self.identity(x)
        gpsa = self.psa(inp)
        res = self.residual(x)
        lu = self.smoothgrad(l)
        y = torch.cat([inp, res, lu, gpsa], dim=1)

        y = self.fusion(y)
        y = self.up(y)
        y = self.rebuild(y)

        return y


if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48).cuda()
    l = torch.rand(1, 3, 48, 48).cuda()

    # model = L0Smooth(20, 64, scale=4).cuda()

    model = L0smooth(20, 64, scale=4, block='RSPA').cuda()

    # flops, params = profile(model, (x, l))
    # print(f'flops: {flops}  params: {params}')
    st = time.time()
    y = model(x, l)
    et = time.time()
    print('time ', et - st)
    print("param_nums  ", compute_num_params(model, False))
    print("output  ", y.shape)


