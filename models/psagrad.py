import math
import torch
import torch.nn as nn
from models import register
from common import normalconv, CALayer, PALayer, Upsampler, compute_num_params, Get_laplacian_gradient
from rcan import RCAB
from smoothx import RSPA
from attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention, SequentialPolarizedSelfAttention
from res2sr import Res2neck,Res2CAPA

@register('psa')
class psa(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2, n_colors=3):
        super().__init__()

        #define identity branch
        m_identity=[]
        m_identity.append(normalconv(n_colors, n_feats))
        m_identity.append(nn.ReLU())
        m_identity.append(normalconv(n_feats, n_feats))
        m_identity.append(Upsampler(normalconv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual=[]
        m_residual.append(normalconv(n_colors, n_feats))
        for _ in range(n_resblocks):
            # m_residual.append(Res2CAPA(n_feats))
            # m_residual.append(RSPA(n_feats))
            m_residual.append(RCAB(normalconv, n_feats, 3, n_feats//4))
            # m_residual.append(SequentialPolarizedSelfAttention(n_feats))
            # m_residual.append(ParallelPolarizedSelfAttention(n_feats))
        m_residual.append(Upsampler(normalconv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)


        m_rebuild=[]
        m_rebuild.append(normalconv(n_feats*2, n_feats))
        # m_rebuild.append(Upsampler(normalconv, scale, n_feats))
        # m_rebuild.append(nn.PReLU(n_feats))
        m_rebuild.append(normalconv(n_feats, n_colors))

        self.rebuild = nn.Sequential(*m_rebuild)

    def forward(self, x):

        inp = self.identity(x)
        # inp = self.up_identity(inp)

        res = self.residual(x)
        # res = self.up_residual(res)

        y = torch.cat([inp, res], dim=1)
        y = self.rebuild(y)

        return y


#融合LR梯度图
@register('psagrad')
class psagrad(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2, n_colors=3):
        super().__init__()

        #define identity branch
        m_identity=[]
        m_identity.append(normalconv(n_colors, n_feats))
        m_identity.append(nn.ReLU())
        m_identity.append(normalconv(n_feats, n_feats))
        # m_identity.append(Upsampler(normalconv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual=[]
        m_residual.append(normalconv(n_colors, n_feats))
        for _ in range(n_resblocks):
            # m_residual.append(Res2CAPA(n_feats))
            m_residual.append(RSPA(n_feats))
            # m_residual.append(RCAB(normalconv, n_feats, 3, n_feats//4))
            # m_residual.append(SequentialPolarizedSelfAttention(n_feats))
            # m_residual.append(ParallelPolarizedSelfAttention(n_feats))
        # m_residual.append(Upsampler(normalconv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        m_grad =[]
        m_grad.append(Get_laplacian_gradient())
        m_grad.append(normalconv(n_colors, n_feats))
        # m_grad.append(Upsampler(normalconv, scale, n_feats))
        self.grad = nn.Sequential(*m_grad)

        m_rebuild=[]
        m_rebuild.append(normalconv(n_feats*3, n_feats))
        m_rebuild.append(Upsampler(normalconv, scale, n_feats))
        m_rebuild.append(nn.PReLU(n_feats))
        m_rebuild.append(normalconv(n_feats, n_colors))

        self.rebuild = nn.Sequential(*m_rebuild)

    def forward(self, x):

        inp = self.identity(x)
        # inp = self.up_identity(inp)

        res = self.residual(x)
        # res = self.up_residual(res)

        grad = self.grad(x)
        # grad = self.up_grad(grad)

        y = torch.cat([inp, res, grad], dim=1)
        y = self.rebuild(y)

        return y


if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48).cuda()
    model = psagrad(n_resblocks=20, n_feats=64, scale=4).cuda()
    y = model(x)
    param_nums = compute_num_params(model)
    print(param_nums)
    print(y.shape)