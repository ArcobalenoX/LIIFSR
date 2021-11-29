import torch
from torchsummary import summary
import torch.nn as nn
 
'''
分组卷积
'''
class groupsconv(nn.Module):
    def __init__(self,in_channel,out_channel,group):
        super(groupsconv,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              groups=group,
                              bias=False)
    def forward(self,input):
        out = self.conv(input)
        return out
 
 
'''
深度可分离卷积
'''
class depthpointconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(depthpointconv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
 
 
 
if __name__=='__main__':
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalconv = nn.Sequential(nn.Conv2d(3,6,kernel_size=3,padding=1,bias=False)).to(device)
    print(summary(normalconv,input_size=(3,32,32)))    

    gc = groupsconv(3,6,3).to(device)

 
    print(summary(gc,input_size=(3,32,32)))
 
    dp = depthpointconv(3,6).to(device)
    print(summary(dp,input_size=(3,32,32)))