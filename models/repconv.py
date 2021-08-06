import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from numpy import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _conv(input_channel, output_channel, kernel_size=(3, 3), padding=(1, 1), stride=1, groups=1, bias=True):
    res = nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                    kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, bias=bias)
    return res


class ACNet(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, groups=1, stride=1, deploy=False):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.deploy = deploy
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.groups = groups
        self.activation = nn.ReLU()

        if(not self.deploy):
            self.brb_3x3 = _conv(
                input_channel, output_channel, kernel_size=3, padding=1, groups=groups)
            self.brb_1x3 = _conv(input_channel, output_channel, kernel_size=(1, 3), padding=(0, 1), groups=groups)
            self.brb_3x1 = _conv(input_channel, output_channel, kernel_size=(3, 1), padding=(1, 0), groups=groups)
        else:
            self.brb_rep = nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                                    kernel_size=self.kernel_size, padding=self.padding, stride=stride, bias=True)

    def forward(self, inputs):
        if(self.deploy):
            return self.activation(self.brb_rep(inputs))
        else:
            return self.activation(self.brb_1x3(inputs)+self.brb_3x1(inputs)+self.brb_3x3(inputs))

    def _switch_to_deploy(self):
        self.deploy = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.brb_rep = nn.Conv2d(in_channels=self.brb_3x3.in_channels, out_channels=self.brb_3x3.out_channels,
                                kernel_size=self.brb_3x3.kernel_size, padding=self.brb_3x3.padding,
                                padding_mode=self.brb_3x3.padding_mode, stride=self.brb_3x3.stride,
                                groups=self.brb_3x3.groups, bias=True)
        self.brb_rep.weight.data = kernel
        self.brb_rep.bias.data = bias
        # 消除梯度更新
        for para in self.parameters():
            para.detach_()
        # 删除没用的分支
        self.__delattr__('brb_3x3')
        self.__delattr__('brb_3x1')
        self.__delattr__('brb_1x3')

    # 将1x3的卷积变成3x3的卷积参数
    def _pad_1x3_kernel(self, kernel):
        if(kernel is None):
            return 0
        else:
            return F.pad(kernel, [0, 0, 1, 1])

    # 将3x1的卷积变成3x3的卷积参数
    def _pad_3x1_kernel(self, kernel):
        if(kernel is None):
           return 0
        else:
           return F.pad(kernel, [1, 1, 0, 0])

    # 将identity，1x1,3x3的卷积融合到一起，变成一个3x3卷积的参数
    def _get_equivalent_kernel_bias(self):
        brb_3x3_weight = self.brb_3x3.weight
        brb_1x3_weight = self.brb_1x3.weight
        brb_3x1_weight = self.brb_3x1.weight
        brb_3x3_bias = self.brb_3x3.bias
        brb_1x3_bias = self.brb_1x3.bias
        brb_3x1_bias = self.brb_3x1.bias
        kernel = brb_3x3_weight + \
            self._pad_1x3_kernel(brb_1x3_weight) + \
                                self._pad_3x1_kernel(brb_3x1_weight)
        bias = brb_3x3_bias+brb_1x3_bias+brb_3x1_bias
        return kernel, bias


class RepBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, groups=1, stride=1, deploy=False):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.deploy = deploy
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.groups = groups
        self.activation = nn.ReLU()

        # make sure kernel_size=3 padding=1
        assert self.kernel_size == 3
        assert self.padding == 1
        if(not self.deploy):
            self.brb_3x3 = _conv(input_channel, output_channel, kernel_size=self.kernel_size,
            padding=self.padding, groups=groups)
            self.brb_1x1 = _conv(input_channel, output_channel,
                                 kernel_size=1, padding=0, groups=groups)
        else:
            self.brb_rep = nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
            kernel_size=self.kernel_size, padding=self.padding, stride=stride, bias=True)

    def forward(self, inputs):
        if(self.deploy):
            return self.activation(self.brb_rep(inputs))
        else:
            return self.activation(self.brb_1x1(inputs)+self.brb_3x3(inputs))

    def _switch_to_deploy(self):
        self.deploy = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.brb_rep = nn.Conv2d(in_channels=self.brb_3x3.in_channels, out_channels=self.brb_3x3.out_channels,
                            kernel_size=self.brb_3x3.kernel_size, padding=self.brb_3x3.padding,
                            padding_mode=self.brb_3x3.padding_mode, stride=self.brb_3x3.stride,
                            groups=self.brb_3x3.groups, bias=True)
        self.brb_rep.weight.data = kernel
        self.brb_rep.bias.data = bias
        # 消除梯度更新
        for para in self.parameters():
            para.detach_()
        # 删除没用的分支
        self.__delattr__('brb_3x3')
        self.__delattr__('brb_1x1')

    # 将1x1的卷积变成3x3的卷积参数
    def _pad_1x1_kernel(self, kernel):
        if(kernel is None):
            return 0
        else:
            return F.pad(kernel, [1]*4)

    # 将identity，1x1,3x3的卷积融合到一起，变成一个3x3卷积的参数
    def _get_equivalent_kernel_bias(self):
        brb_3x3_weight = self.brb_3x3.weight
        brb_1x1_weight = self.brb_1x1.weight
        brb_3x3_bias = self.brb_3x3.bias
        brb_1x1_bias = self.brb_1x1.bias
        kernel = brb_3x3_weight+self._pad_1x1_kernel(brb_1x1_weight)
        bias = brb_3x3_bias+brb_1x1_bias
        return kernel, bias





class repconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, groups=1, stride=1, deploy=False):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.deploy = deploy
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.groups = groups
        self.activation = nn.ReLU()

        if(not self.deploy):
            self.brb_3x3 = _conv(
                input_channel, output_channel, kernel_size=3, padding=1, groups=groups)
            self.brb_1x3 = _conv(input_channel, output_channel, kernel_size=(1, 3), padding=(0, 1), groups=groups)
            self.brb_3x1 = _conv(input_channel, output_channel, kernel_size=(3, 1), padding=(1, 0), groups=groups)
            self.brb_1x1 = _conv(
                input_channel, output_channel, kernel_size=1, padding=0, groups=groups)
        else:
            self.brb_rep = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, 
            kernel_size=self.kernel_size, padding=self.padding, stride=stride, bias=True)

    def forward(self, inputs):
        if(self.deploy):
            return self.activation(self.brb_rep(inputs))
        else:
            x = self.brb_1x3(inputs)+self.brb_3x1(inputs) + \
                            self.brb_3x3(inputs)+self.brb_1x1(inputs)
            return self.activation(x)

    def _switch_to_deploy(self):
        self.deploy = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.brb_rep = nn.Conv2d(in_channels=self.brb_3x3.in_channels, out_channels=self.brb_3x3.out_channels,
                                kernel_size=self.brb_3x3.kernel_size, padding=self.brb_3x3.padding,
                                padding_mode=self.brb_3x3.padding_mode, stride=self.brb_3x3.stride,
                                groups=self.brb_3x3.groups, bias=True)
        self.brb_rep.weight.data = kernel
        self.brb_rep.bias.data = bias
        # 消除梯度更新
        for para in self.parameters():
            para.detach_()
        # 删除没用的分支
        self.__delattr__('brb_3x3')
        self.__delattr__('brb_3x1')
        self.__delattr__('brb_1x3')
        self.__delattr__('brb_1x1')

    # 将1x3的卷积变成3x3的卷积参数
    def _pad_1x3_kernel(self, kernel):
        if(kernel is None):
            return 0
        else:
            return F.pad(kernel, [0, 0, 1, 1])

    # 将3x1的卷积变成3x3的卷积参数
    def _pad_3x1_kernel(self, kernel):
        if(kernel is None):
            return 0
        else:
            return F.pad(kernel, [1, 1, 0, 0])
    # 将1x1的卷积变成3x3的卷积参数
    def _pad_1x1_kernel(self,kernel):
        if(kernel is None):
            return 0
        else:
            return F.pad(kernel,[1]*4)
    

    # 将1x1,3x3,1x3,3x1的卷积融合到一起，变成一个3x3卷积的参数
    def _get_equivalent_kernel_bias(self):
        brb_3x3_weight = self.brb_3x3.weight
        brb_1x3_weight = self._pad_1x3_kernel(self.brb_1x3.weight)
        brb_3x1_weight = self._pad_3x1_kernel(self.brb_3x1.weight)
        brb_1x1_weight = self._pad_1x1_kernel(self.brb_1x1.weight)
        brb_3x3_bias = self.brb_3x3.bias
        brb_1x3_bias = self.brb_1x3.bias
        brb_3x1_bias = self.brb_3x1.bias
        brb_1x1_bias = self.brb_1x1.bias          
        kernel = brb_3x3_weight+brb_1x3_weight+brb_3x1_weight+brb_1x1_weight
        bias = brb_3x3_bias+brb_1x3_bias+brb_3x1_bias+brb_1x1_bias
        return kernel,bias


def acnet_test():
    input=torch.randn(1,32,49,49)
    acnet=ACNet(32,32)
    acnet.eval()
    out=acnet(input)
    acnet._switch_to_deploy()
    out2=acnet(input)
    print('difference:')
    print(((out2-out)**2).sum())


def repvgg_test():
    input=torch.randn(1,32,49,49)
    repblock=RepBlock(32,32)
    repblock.eval()
    out=repblock(input)
    repblock._switch_to_deploy()
    out2=repblock(input)
    print('difference between vgg and repvgg')
    print(((out2-out)**2).sum())    


def repconv_test():
    input=torch.randn(1,32,49,49)
    repnet=repconv(32,32)
    repnet.eval()
    out=repnet(input)
    repnet._switch_to_deploy()
    out2=repnet(input)
    print('difference:')
    print(((out2-out)**2).sum())

if __name__ == '__main__':
    acnet_test()
    repvgg_test() 
    repconv_test()   
