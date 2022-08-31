import math
import torch
import torch.nn as nn
from models import register
from common import conv, PALayer, Upsampler, compute_num_params
from attention.CBAM import SpatialAttention



class RDAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=(3//2))
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=(3//2))
        self.conv3 = nn.Conv2d(n_feats*3, n_feats, 1, padding=(1//2))
        self.pa = PALayer(n_feats, 8)
        self.sa = SpatialAttention(3)

    def forward(self, x):
        iden = x
        x1 = self.conv1(x)
        x2 = self.act1(x1)
        x2 = self.conv2(x2)
        xm = torch.cat([x1, x2,  iden], dim=1)
        xm = self.conv3(xm)
        p = self.pa(xm)
        s = self.sa(xm)
        #y = p + s
        y = p+s+iden
        return y

@register('dasr')
class DASR(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2):
        super().__init__()
        n_colors = 3

        #define identity branch 全局残差分支
        m_identity = []
        m_identity.append(conv(n_colors, n_feats))
        m_identity.append(Upsampler(conv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)
        self.identity_up = conv(n_feats, n_colors)

        # define residual branch
        m_residual = []
        m_residual.append(conv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(RDAB(n_feats))
        m_residual.append(Upsampler(conv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)
        self.residual_up = conv(n_feats, n_colors)


    def forward(self, x):
        inp_up = self.identity_up(self.identity(x))
        #print(inp_up.shape)
        res_up = self.residual_up(self.residual(x))
        #print(res_up.shape)
        y = inp_up + res_up
        return y



if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48).cuda()
    model = DASR(n_resblocks=10, n_feats=64, scale=2).cuda()
    y = model(x)
    #print(model)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)

