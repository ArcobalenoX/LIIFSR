import math
import torch
import torch.nn as nn
from models import register
from common import conv, CALayer, PALayer, Upsampler, compute_num_params, Get_laplacian_gradient, RCAB


class ParallelPolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).\
            permute(0, 2, 1).reshape(b, c, 1, 1)  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out


class SequentialPolarizedSelfAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).\
            permute(0, 2,1).reshape(b, c, 1, 1)  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(channel_out)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(channel_out)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * channel_out
        return spatial_out



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
    def __init__(self, n_feats, reduction=8):
        super().__init__()
        self.conv = Res2neck(n_feats, n_feats)
        self.ca = CALayer(n_feats, reduction)
        self.pa = PALayer(n_feats, reduction)

    def forward(self, x):
        y = self.conv(x)+x
        y = self.ca(y)+x
        y = self.pa(y)+x
        y = y+x
        return y


#融合LR梯度图
@register('psagrad')
class psagrad(nn.Module):
    def __init__(self, n_resblocks=20, n_feats=64, scale=2, n_colors=3):
        super().__init__()

        #define identity branch
        m_identity=[]
        m_identity.append(conv(n_colors, n_feats))
        # m_identity.append(nn.ReLU())
        # m_identity.append(conv(n_feats, n_feats))
        # m_identity.append(Upsampler(conv, scale, n_feats))
        self.identity = nn.Sequential(*m_identity)

        # define residual branch
        m_residual=[]
        m_residual.append(conv(n_colors, n_feats))
        for _ in range(n_resblocks):
            # m_residual.append(Res2neck(n_feats,n_feats))
            m_residual.append(RCAB(conv, n_feats, 3, n_feats//4))
            # m_residual.append(SequentialPolarizedSelfAttention(n_feats))
            # m_residual.append(ParallelPolarizedSelfAttention(n_feats))
        # m_residual.append(Upsampler(conv, scale, n_feats))
        self.residual = nn.Sequential(*m_residual)

        m_grad =[]
        m_grad.append(Get_laplacian_gradient())
        m_grad.append(conv(n_colors, n_feats))
        # m_grad.append(Upsampler(conv, scale, n_feats))
        self.grad = nn.Sequential(*m_grad)

        m_rebuild=[]
        m_rebuild.append(conv(n_feats*3, n_feats))
        m_rebuild.append(Upsampler(conv, scale, n_feats))
        m_rebuild.append(nn.PReLU(n_feats))
        m_rebuild.append(conv(n_feats, n_colors))

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




