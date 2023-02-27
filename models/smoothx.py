import torch
import torch.nn as nn
from models import register
from common import compute_num_params, normalconv, Upsampler, PALayer, CALayer, SAM

#小论文使用
class RSPA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = normalconv(n_feats, n_feats, 3)
        self.act1 = nn.ReLU(True)
        self.conv2 = normalconv(n_feats, n_feats, 3)
        self.conv3 = normalconv(n_feats*3, n_feats, 1)
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

@register('RSPAL0')
class RSPAL0(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()
        kernel_size = 3
        n_colors = 3
        #define identity branch
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

        self.fusion = normalconv(n_feats*3, 3, kernel_size)


    def forward(self, x, l):
        inp = self.identity(x)
        lu = self.smoothgrad(l)
        res = self.residual(x)
        y = torch.cat([inp, lu, res], dim=1)
        y = self.fusion(y)
        return y


@register('L0Smoothsamx')
class L0Smooth(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()
        kernel_size = 3
        n_colors = 3
        #define identity branch
        m_identity = []
        m_identity.append(normalconv(n_colors, n_feats))
        m_identity.append(Upsampler(normalconv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)
        self.identity_up = normalconv(n_feats, n_colors)

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

        self.fusion = normalconv(n_feats*3, 3, kernel_size)

        # self.sam = SAM(n_colors, kernel_size, True)
        # self.gradup = Upsampler(normalconv, scale, n_feats)

    def forward(self, x, l):
        inp = self.identity(x)
        lu = self.smoothgrad(l)
        res = self.residual(x)
        y = torch.cat([inp, lu, res], dim=1)
        y = self.fusion(y)
        return y


class MKRB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.brb_3x3 = nn.Conv2d(n_feats, n_feats//4, kernel_size=3, padding=1)
        self.brb_1x3 = nn.Conv2d(n_feats, n_feats//4, kernel_size=(1, 3), padding=(0, 1))
        self.brb_3x1 = nn.Conv2d(n_feats, n_feats//4, kernel_size=(3, 1), padding=(1, 0))
        self.brb_1x1 = nn.Conv2d(n_feats, n_feats//4, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(0.01, True)

        self.conv2 = normalconv(n_feats, n_feats, 3)
        self.conv3 = normalconv(n_feats*3, n_feats, 1)
        self.att = PALayer(n_feats, n_feats//4)

    def forward(self, x):

        x33 = self.brb_3x3(x)
        x13 = self.brb_1x3(x)
        x31 = self.brb_3x1(x)
        x11 = self.brb_1x1(x)
        xm = torch.cat([x33, x13, x31, x11], dim=1)
        xm = self.act(xm)

        x1 = x
        x2 = xm
        x3 = self.conv2(x2)
        y = torch.cat([x3, x2, x1], dim=1)
        y = self.conv3(y)
        y = self.att(y)+x
        
        return y

#小论文参数量
class MKRA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.brb_3x3 = nn.Conv2d(n_feats, n_feats//4, kernel_size=3, padding=1)
        self.brb_1x3 = nn.Conv2d(n_feats, n_feats//4, kernel_size=(1, 3), padding=(0, 1))
        self.brb_3x1 = nn.Conv2d(n_feats, n_feats//4, kernel_size=(3, 1), padding=(1, 0))
        self.brb_1x1 = nn.Conv2d(n_feats, n_feats//4, kernel_size=1, padding=0)
        self.act = nn.ReLU(True)
        self.ca = CALayer(n_feats, 8)
        self.pa = PALayer(n_feats, 8)

    def forward(self, x):
        iden = x
        x33 = self.brb_3x3(x)
        x13 = self.brb_1x3(x)
        x31 = self.brb_3x1(x)
        x11 = self.brb_1x1(x)
        xm = torch.cat([x33, x13, x31, x11], dim=1)
        xm = self.act(xm)
        y = self.ca(xm+iden)
        y = self.pa(y)
        y = y+iden
        return y

@register('mkran')
class MKRAN(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()
        n_colors = 3
        #define identity branch
        m_identity = []
        m_identity.append(normalconv(n_colors, n_feats))
        m_identity.append(Upsampler(normalconv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(normalconv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(MKRA(n_feats))
        m_residual.append(Upsampler(normalconv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        # define L0 grad branch
        m_smoothgrad = []
        m_smoothgrad.append(normalconv(n_colors, n_feats))
        m_smoothgrad.append(Upsampler(normalconv, scale, n_feats))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.fusion = normalconv(n_feats*3, n_colors)

    def forward(self, x, l0):
        inp = self.identity(x)
        res = self.residual(x)
        l0grad = self.smoothgrad(l0)
        y = torch.cat([inp, res, l0grad], dim=1)
        y = self.fusion(y)
        return y

@register('L0mksam')
class L0mksam(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()
        kernel_size = 3
        n_colors = 3
        #define identity branch
        m_identity = []
        m_identity.append(normalconv(n_colors, n_feats))
        m_identity.append(Upsampler(normalconv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(normalconv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(MKRB(n_feats))
        m_residual.append(Upsampler(normalconv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        # define grad branch
        m_smoothgrad = []
        m_smoothgrad.append(Upsampler(normalconv, scale, n_colors))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.fusion = normalconv(n_feats, 3, kernel_size)
        self.sam = SAM(n_feats, kernel_size, True)

    def forward(self, x, l):
        inp = self.identity(x)
        res = self.residual(x)
        lu = self.smoothgrad(l)
        deep = inp+res
        y, img = self.sam(deep, lu)
        y = self.fusion(y)
        return y

    
@register('L0mkrspa')
class L0mkrspa(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2, n_colors = 3):
        super().__init__()

        #define identity branch
        m_identity = []
        m_identity.append(normalconv(n_colors, n_feats))
        m_identity.append(Upsampler(normalconv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(normalconv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(MKRB(n_feats))
        m_residual.append(Upsampler(normalconv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        # define grad branch
        m_smoothgrad = []
        m_smoothgrad.append(normalconv(n_colors, n_feats))
        m_smoothgrad.append(Upsampler(normalconv, scale, n_feats))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.fusion = normalconv(n_feats*3, n_colors)


    def forward(self, x, l):
        inp = self.identity(x)
        res = self.residual(x)
        lu = self.smoothgrad(l)
        y = torch.cat([inp, res, lu], dim=1)
        y = self.fusion(y)
        return y    
    

if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48).cuda()
    l = torch.rand(1, 3, 48, 48).cuda()
    model_name = L0mksam
    model = model_name(15, 48, scale=4).cuda()
    y = model(x, l)
    print("param_nums  ", compute_num_params(model, False))
    print("outpute  ", y.shape)

    model = MKRAN(n_resblocks=20, n_feats=64, scale=4).cuda()
    y = model(x, l)
    #print(model)
    print("param_nums  ", compute_num_params(model, False))
    print("outpute  ", y.shape)

