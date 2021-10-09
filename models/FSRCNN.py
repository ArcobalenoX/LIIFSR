import torch
import torch.nn as nn
import torch.nn.functional as F


class FSRCNN_net(nn.Module):
    def __init__(self, input_channels, upscale, d=64, s=12, m=4):
        super().__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=d, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.tail_conv = nn.ConvTranspose2d(in_channels=d, out_channels=input_channels, kernel_size=9,
                                            stride=upscale, padding=3, output_padding=1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        print(fea.shape)
        out = self.tail_conv(fea)
        return out

if __name__ == '__main__':
    x = torch.rand(1, 3, 128, 128)
    model = FSRCNN_net(input_channels=3, upscale=2)
    y = model(x)
    print(model)
    #param_nums = utils.compute_num_params(model)
    #print(param_nums)
    print(y.shape)
