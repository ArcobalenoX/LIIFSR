import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register
from common import normalconv, compute_num_params

@register('lgcnet')
class LGCNET(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.conv1 = normalconv(3, 32)
        self.conv2 = normalconv(32, 32)
        self.conv3 = normalconv(32, 32)
        self.conv4 = normalconv(32, 32)
        self.conv5 = normalconv(32, 32)
        self.conv6 = normalconv(96, 64, 5)
        self.conv7 = normalconv(64, 3)

    def forward(self, x):

        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)

        f6 = torch.cat([f3, f4, f5], dim=1)
        f6 = self.conv6(f6)
        f7 = self.conv7(f6)
        y = f7+x

        return y



if __name__ == '__main__':
    x = torch.rand(1, 3, 48, 48)
    model = LGCNET(scale=8)
    y = model(x)
    print(model)
    param_nums = compute_num_params(model,False)
    print(param_nums)
    print(y.shape)

