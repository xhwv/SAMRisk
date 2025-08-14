import torch.nn as nn
import torch
from torch.nn import init, Parameter

class ChannelAttention3D(nn.Module):
    def __init__(self, channel, reduction_ratio=2):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction_ratio, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channel * reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class conv_block(nn.Module):
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        self.dw_conv = nn.Conv3d(c_in, c_in, kernel_size=k_size, groups=c_in, stride=1,
                                 padding=(k_size - 1) // 2, bias=False)

        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        self.conv1=nn.Conv3d(c_in*scale,c_in,kernel_size=(1,1,1),stride=1)
        self.act = nn.LeakyReLU()
        self.compress = nn.Conv3d(c_in, c_in//scale, kernel_size=(1, 1, 1), stride=1)
        self.conv2=nn.Conv3d(c_in//scale,c_in,kernel_size=(1,1,1),stride=1)
        self.norm=nn.BatchNorm3d(c_in)
        self.drop=nn.Dropout(0.1)
        self.conv111=nn.Conv3d(c_in, c_in, kernel_size=(1, 1, 1), stride=1)
        self.conv222 = nn.Conv3d(c_in, c_in, kernel_size=(1, 1, 1), stride=1)
        self.catt=ChannelAttention3D(c_in)
        self.bn=nn.BatchNorm3d(c_in)

    def forward(self, x):
        x=self.dw_conv(x)
        x=self.catt(x)
        a=x
        b=x
        a=self.expansion(a)
        a=self.act(a)
        a=self.drop(a)
        a=self.conv1(a)
        b=self.compress(b)
        b=self.act(b)
        b=self.drop(b)
        b=self.conv2(b)
        x=a+b
        x=self.bn(x)
        shortcut=x
        x=self.conv111(x)
        x = self.act(x)
        x = self.drop(x)
        x=self.conv222(x)
        x=x+shortcut
        return x

class FR(nn.Module):
    def __init__(self, in_channel, base_c, k_size, num_block, scale, num_class):
        super().__init__()

        self.layer1 = self._make_layer(base_c, num_block[0], scale[0], k_size)
        self.down = nn.Conv3d(base_c, base_c, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.layer2 = self._make_layer(base_c , num_block[1], scale[1], k_size)
        self.po=nn.AdaptiveAvgPool3d((1, 1, 1))
        self.sig = nn.Sigmoid()
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        self.drop = nn.Dropout(0.1)
        self.re=nn.LeakyReLU()
        self.fcc1=nn.Linear(384,64)
        self.fcc2=nn.Linear(64,1)
        self.bn=nn.BatchNorm3d(384)
        self.bat1=nn.BatchNorm3d(384)
        self.bat2=nn.BatchNorm1d(384)

    def _make_layer(self, c_in, n_conv, ratio, k_size):
        layers = []
        for _ in range(n_conv):
            layers.append(conv_block(c_in, ratio, k_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.bn(x)
        out1 = self.layer1(x)
        d1 = self.down(out1)
        d1=self.bat1(d1)
        d1 = self.re(d1)
        d1 = self.drop(d1)
        out2 = self.layer2(d1)
        out=self.po(out2)
        out = torch.flatten(out, start_dim=1)
        out=self.bat2(out)
        out=self.fcc1(out)
        out = self.re(out)
        out = self.drop(out)
        out=self.fcc2(out)
        out = self.sig(out)
        out = out * self.output_range + self.output_shift
        return out

def get_mednet():
    num_block = [1,1,1]
    scale = [4, 4,4]
    net = FR(384, 384, 3, num_block, scale, 1)
    return net