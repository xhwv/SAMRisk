import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from spconv import pytorch as spconv


class GlobalSparseAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.1,
        sr_ratio=2,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool3d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = nn.ConvTranspose3d(
                dim, dim, kernel_size, stride=sr_ratio, groups=dim
            )
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()
        # self.ln=nn.LayerNorm(dim)

    def forward(self, x, D: int, H: int, W: int):
        B, N, C = x.shape

        if self.sr > 1.0:
            x = x.transpose(1, 2).reshape(B, C, D, H, W)
            shortcut=x
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )


        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(D / self.sr), int(H / self.sr), int(W / self.sr))
            x = self.LocalProp(x)
            x=x+shortcut
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        # x=self.ln(x)
        x = self.proj_drop(x)
        return x
class SelfAttn(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=2.0,
    ):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, D, H, W)
        return x
class LMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.ln=nn.LayerNorm(out_features)
        # self.l=nn.LayerNorm(hidden_features)

    def forward(self, x):
        x = self.fc1(x)
        # x=self.l(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x=self.ln(x)
        return x

class GLNLBlock(nn.Module):
    # def __init__(self, dim,k=3,s=1,p=1):
    def __init__(self, dim, k=7, s=1, p=3):
        super(GLNLBlock, self).__init__()
        self.fc=nn.Linear(dim,dim*4)
        self.fc1=nn.Linear(dim*4,dim)
        self.conv=nn.Conv3d(dim,dim*4,kernel_size=k,stride=s,padding=p)
        self.conv1=nn.Conv3d(dim*4,dim,kernel_size=k,stride=s,padding=p)
        self.sp=SelfAttn(dim=4,num_heads=4,sr_ratio=4)
        self.bat5 = nn.BatchNorm3d(dim)
        self.drop=nn.Dropout(0.1)
        self.relu=nn.LeakyReLU()
    def forward(self, x):
        # x=self.bat1(x)
        shortcut=x
        x=self.sp(x)
        # x=self.bat2(x)
        a=x.permute(0, 2, 3, 4, 1)
        a=self.fc(a)
        a=self.relu(a)
        a=self.drop(a)
        a=self.fc1(a)
        a=a.permute(0, 4, 1, 2, 3)
        b=self.conv(x)
        # b=self.bn(b)
        b=self.relu(b)
        b=self.drop(b)
        b=self.conv1(b)
        x=a+b
        x=x+shortcut
        x=self.bat5(x)
        return x

class GLNLBlock5(nn.Module):
    def __init__(self, dim,k=5,s=1,p=2):
        super(GLNLBlock5, self).__init__()
        self.fc=nn.Linear(dim,dim*4)
        self.fc1=nn.Linear(dim*4,dim)
        self.conv=nn.Conv3d(dim,dim*4,kernel_size=k,stride=s,padding=p)
        self.conv1 = nn.Conv3d(dim*4, dim, kernel_size=k, stride=s, padding=p)
        self.sp=SelfAttn(dim=4,num_heads=4,sr_ratio=2)
        self.bat2 = nn.BatchNorm3d(dim)
        self.drop=nn.Dropout(0.1)
        self.relu=nn.LeakyReLU()
    def forward(self, x):
        shortcut=x
        x = self.sp(x)
        a=x.permute(0, 2, 3, 4, 1)
        a=self.fc(a)
        a=self.relu(a)
        a=self.drop(a)
        a=self.fc1(a)
        a=a.permute(0, 4, 1, 2, 3)
        b=self.conv(x)
        b=self.relu(b)
        b=self.drop(b)
        b=self.conv1(b)
        x=a+b
        x = x + shortcut
        x=self.bat2(x)
        return x
class GLNLBlock7(nn.Module):
    # def __init__(self, dim,k=7,s=1,p=3):
    def __init__(self, dim, k=3, s=1, p=1):
        super(GLNLBlock7, self).__init__()
        self.fc=nn.Linear(dim,dim*4)
        self.fc1=nn.Linear(dim*4,dim)
        self.conv=nn.Conv3d(dim,dim*4,kernel_size=k,stride=s,padding=p)
        self.conv1 = nn.Conv3d(dim*4, dim, kernel_size=k, stride=s, padding=p)
        self.sp=SelfAttn(dim=4,num_heads=4,sr_ratio=2)
        self.bat4 = nn.BatchNorm3d(dim)
        self.drop = nn.Dropout(0.1)
        self.relu=nn.LeakyReLU()
    def forward(self, x):
        shortcut = x
        x = self.sp(x)
        a=x.permute(0, 2, 3, 4, 1)
        a=self.fc(a)
        a=self.relu(a)
        a=self.drop(a)
        a=self.fc1(a)
        a=a.permute(0, 4, 1, 2, 3)
        b=self.conv(x)
        b=self.relu(b)
        b=self.drop(b)
        b=self.conv1(b)
        x=a+b
        x = x + shortcut
        x=self.bat4(x)
        return x




class MVRF(nn.Module):
    def __init__(self, dim=4):
        super(earlyfusion, self).__init__()
        self.dcd=GLNLBlock(dim)
        # self.dcdy=dcdBlock(dim)
        self.dcd5=GLNLBlock5(dim)
        self.dcd7=GLNLBlock7(dim)
        self.LocalProp = nn.ConvTranspose3d(
            dim, dim, kernel_size=4, stride=4, groups=dim
        )
        self.avg_p = nn.AvgPool3d(kernel_size=4, stride=4)
        self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.trans_conv1 = nn.ConvTranspose3d(in_channels=4, out_channels=4,
                                              kernel_size=(4, 4, 4),
                                              stride=(2, 2, 2),
                                              padding=(1, 1, 1))
        self.trans_conv2 = nn.ConvTranspose3d(in_channels=4, out_channels=4,
                                              kernel_size=(4, 4, 4),
                                              stride=(2, 2, 2),
                                              padding=(1, 1, 1))
        self.trans_conv3 = nn.ConvTranspose3d(in_channels=4, out_channels=4,
                                              kernel_size=(4, 4, 4),
                                              stride=(2, 2, 2),
                                              padding=(1, 1, 1))
        self.largeconv=nn.Conv3d(1,4,kernel_size=7,stride=1,padding=3)
        self.down3=DownsampleConv3d16(1)
        self.down2=DownsampleConv3d32(1)
        self.down1=nn.Conv3d(1, 4, kernel_size=3, stride=2, padding=1)
        self.bat = nn.BatchNorm3d(dim)
        self.re=nn.LeakyReLU()
        self.drop=nn.Dropout(0.1)

    def forward(self, x,y):
        short=x
        shorty=y.repeat(1, 4, 1, 1, 1)
        y3 = self.down3(y)
        y2 = self.down2(y)
        y1 = self.down1(y)
        xp1=self.avg_pool(x)
        xp1=xp1+y1
        x1=self.dcd(xp1)
        xp2=self.avg_pool(xp1)
        xp2=xp2+y2
        x2=self.dcd5(xp2)
        xp3=self.avg_pool(xp2)
        xp3=xp3+y3
        x3=self.dcd7(xp3)
        x3=self.drop(x3)
        x3=self.re(x3)
        x3=self.trans_conv3(x3)
        x2=x2+x3
        x2=self.drop(x2)
        x2=self.re(x2)
        x2=self.trans_conv2(x2)
        x1=x2+x1
        x1=self.drop(x1)
        x1=self.re(x1)
        x=self.trans_conv1(x1)
        x=x+short+shorty
        x=self.bat(x)
        return x

class DownsampleConv3d16(nn.Module):
    def __init__(self, channels):
        super(DownsampleConv3d16, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channels, channels*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(channels*2, channels*4, kernel_size=3, stride=2, padding=1)
        self.re=nn.LeakyReLU()
        self.drop=nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)  # 降采样到 64
        x=self.re(x)
        x=self.drop(x)
        x = self.conv2(x)  # 降采样到 32
        x = self.re(x)
        x = self.drop(x)
        x = self.conv3(x)  # 降采样到 16

        return x


class DownsampleConv3d32(nn.Module):
    def __init__(self, channels):
        super(DownsampleConv3d32, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channels, channels*4, kernel_size=3, stride=2, padding=1)
        self.re = nn.LeakyReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)  # 降采样到 64
        x=self.re(x)
        x=self.drop(x)
        x = self.conv2(x)  # 降采样到 32
        return x
