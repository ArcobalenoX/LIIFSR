import torch.nn as nn

from models import register


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #print(self.layers)
        #print("f",x.shape)#[N*h*w,?]
        shape = x.shape[:-1]
        #print("s",shape)
        x = x.view(-1, x.shape[-1])
        #print("g",x.shape)
        x = self.layers(x)
        #print("h", x.shape)#[N*h*w,3]
        return x.view(*shape, -1)
