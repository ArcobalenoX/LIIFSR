import torch
import torch.nn as nn
from models import register
from common import normalconv, SELayer, Upsampler, compute_num_params

class RSEB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = normalconv(n_feats, n_feats, 3)
        self.act1 = nn.ReLU(True)
        self.conv2 = normalconv(n_feats, n_feats//2, 3)
        self.conv3 = normalconv(n_feats+n_feats+n_feats//2, n_feats, 1)
        self.se = SELayer(n_feats, 8)

    def forward(self, x):
        x1 = x
        x2 = self.conv1(x1)
        x2 = self.act1(x2)
        x3 = self.conv2(x2)
        y = torch.cat([x3, x2, x1], dim=1)
        y = self.conv3(y)
        y = self.se(y)+x
        return y


@register('drsens')
class DRSEN(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2, upsampling=True):
        super().__init__()

        kernel_size = 3
        n_colors = 3
        act = False
        #act = "prelu"

        #define identity branch
        m_identity = []
        m_identity.append(Upsampler(normalconv, scale, n_colors, act=act))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual = []
        m_residual.append(normalconv(n_colors, n_feats))
        for _ in range(n_resblocks):
            m_residual.append(RSEB(n_feats))
        m_residual.append(normalconv(n_feats, n_colors, kernel_size))
        m_residual.append(Upsampler(normalconv, scale, n_colors, act=act))
        self.residual = nn.Sequential(*m_residual)

    def forward(self, x):
        inp = self.identity(x)
        res = self.residual(x)
        y = res+inp
        return y



if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48)
    model = DRSEN(scale=8)
    y = model(x)
    #print(model)
    param_nums = compute_num_params(model, False)
    print(param_nums)
    print(y.shape)

