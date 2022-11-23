import math
import torch
import torch.nn as nn
from models import register
from common import conv, CALayer, PALayer, Upsampler, compute_num_params, Get_laplacian_gradient


class Res2neck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, baseWidth=26, scale = 4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super().__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)

        self.nums = scale-1 if scale !=1 else 1
        convs = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = 1, padding=1, bias=False))
        self.convs = nn.ModuleList(convs)

        self.conv3 = nn.Conv2d(width*scale, planes, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          sp = spx[i] if i==0 else sp + spx[i]
          sp = self.relu(self.convs[i](sp))
          out = sp if i==0 else torch.cat((out, sp), 1)

        if self.scale != 1 :
          out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out += residual
        out = self.relu(out)

        return out


class Res2CAPA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv = Res2neck(n_feats, n_feats)
        self.ca = CALayer(n_feats, 8)
        self.pa = PALayer(n_feats, 8)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = self.ca(y)
        y = self.pa(y)
        y = y+residual
        return y

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

#融合LR梯度图
@register('res2cpgrad')
class res2cpgrad(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2, n_colors=3 ):
        super().__init__()

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
            m_residual.append(Res2CAPA(n_feats))
        m_residual.append(Upsampler(conv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)
        self.residual_up = conv(n_feats, n_colors)

        self.get_grad = Get_laplacian_gradient()
        self.grad_up = Upsampler(conv, scale, n_colors)

        #self.fusion = conv(n_colors, n_colors)


    def forward(self, x):

        inp = self.identity_up(self.identity(x))
        #print(inp.shape)

        grad = self.get_grad(x)
        grad_up = self.grad_up(grad)
        #print("grad", grad.shape)
        #print("grad_up", grad_up.shape)

        res = self.residual(x)
        res_up = self.residual_up(res)
        #print("res", res.shape)
        #print("res_up", res_up.shape)

        y = inp + res_up + grad_up
        #y = self.fusion(y)
        return y


#融合L0梯度图
@register('res2l0')
class res2l0(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2, n_colors=3):
        super().__init__()

        #define identity branch
        m_identity = []
        m_identity.append(conv(n_colors, n_feats))
        m_identity.append(Upsampler(conv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(conv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(Res2neck(n_feats, n_feats))
        m_residual.append(Upsampler(conv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        # define grad branch
        m_smoothgrad = []
        m_smoothgrad.append(conv(n_colors, n_feats))
        m_smoothgrad.append(Upsampler(conv, scale, n_feats))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.fusion = conv(n_feats*3, n_colors)


    def forward(self, x, l):
        inp = self.identity(x)
        lu = self.smoothgrad(l)
        res = self.residual(x)
        y = torch.cat([inp, lu, res], dim=1)
        y = self.fusion(y)
        return y


if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48).cuda()
    model = res2cpgrad(n_resblocks=20, n_feats=64, scale=4).cuda()
    y = model(x)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)

    l = torch.rand(1, 3, 48, 48).cuda()
    model = res2l0(n_resblocks=20, n_feats=64, scale=4).cuda()
    y = model(x, l)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)


