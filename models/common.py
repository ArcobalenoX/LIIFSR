import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_num_params(model, text=True):
    #tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    tot = sum([p.nelement() for p in model.parameters()])
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def compute_flops(model, images):
    """
        images like torch.rand(1, 3, 224, 224)
    """
    from thop import profile
    flops, params = profile(model, (images,))
    print(f'flops: {flops}  params: {params}')


def conv(in_channels, out_channels, kernel_size=3, bias=True, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride=stride)


## Residual  Block (RB)
class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        self.conv = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                  act,
                                  conv(n_feat, n_feat, kernel_size, bias=bias)
                                  )
        self.res_scale = res_scale

    def forward(self, x):
        y = self.conv(x)
        y = y * self.res_scale + x
        return y


#分组卷积
class groupsconv(nn.Module):
    def __init__(self, in_channel, out_channel, group):
        super(groupsconv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, groups=group)
    def forward(self, input):
        out = self.conv(input)
        return out


#深度点卷积
class depthpointconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(depthpointconv, self).__init__()
        self.depth_conv =  nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, groups=1)
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, act=None, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super().__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, _, h, w = x.shape
        y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_ave = self.conv_du(y_ave)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return x*y_ave


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        #res = self.body(x)
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super().__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class PALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, 1, kernel_size=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class FFA(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.conv1 = conv(channel, channel)
        self.act1 = nn.ReLU(True)
        self.conv2 = conv(channel, channel)
        self.ca = CALayer(channel, reduction)
        self.pa = PALayer(channel, reduction)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res += x
        res = self.conv2(res)
        res = self.ca(res)
        res = self.pa(res)
        res += x
        return res


# SELayer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# SEResBlock
class SEResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(SELayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        #res = self.body(x)
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super().__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super().__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class Get_gradient(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x


class Get_sobel_gradient(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x


class Get_laplacian_gradient(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = F.conv2d(x[:, i].unsqueeze(1), self.weight, padding=1)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x
