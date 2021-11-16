import torch
import torch.nn as nn
from models import register
from common import compute_num_params, conv, SELayer, Upsampler, SAM, PALayer, Get_gradient_nopadding, FFA

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
    def __init__(self, n_resblocks=5, n_feats=32, scale=2):
        super().__init__()

        kernel_size = 3
        n_colors = 3

        self.identity_up = conv(n_feats, n_colors)
        self.identity_sr = Upsampler(conv, scale, n_colors)

        #define identity branch
        m_identity = []
        m_identity.append(conv(n_colors, n_feats))
        m_identity.append(Upsampler(conv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)




        # define residual branch
        m_residual = []
        m_residual.append(conv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(RSPA(n_feats))
        m_residual.append(Upsampler(conv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        m_smoothgrad = []
        m_smoothgrad.append(conv(n_colors, n_feats))
        m_smoothgrad.append(Upsampler(conv, scale, n_feats))
        self.smoothgrad = nn.Sequential(*m_smoothgrad)

        self.sam = SAM(n_colors, kernel_size, True)

        self.grad = Get_gradient_nopadding()
        self.gradup = Upsampler(conv, scale, n_colors)

        self.ffa = FFA(n_feats*3)
        self.fusion = conv(n_feats*3, 3, kernel_size)


    def forward(self, x, l):

        x_up = self.identity_sr(x)
        x_grad = self.grad(x)
        x_grad_up = self.gradup(x_grad)
        print(x_grad.shape)

        xf_up = self.identity(x)
        lf_up = self.smoothgrad(l)
        res = self.residual(x)

        y = torch.cat([xf_up, lf_up, res], dim=1)
        y = self.ffa(y)
        y = self.fusion(y)
        y = y+x_up+x_grad_up

        return y


if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48).cuda()
    l = torch.rand(1, 3, 48, 48).cuda()
    model = L0SmoothSR(scale=4).cuda()
    y = model(x, l)
    print(model)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)

