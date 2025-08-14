import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch.nn import init, Parameter

class CMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.bat=nn.BatchNorm3d(out_features)
        # self.bn=nn.BatchNorm3d(hidden_features)

    def forward(self, x):
        x = self.fc1(x)
        # x=self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x=self.bat(x)
        return x
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
        # self.ln=nn.LayerNorm(dim)

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
class LMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.LeakyReLU,
        drop=0.1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x=self.ln(x)
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
        act_layer=nn.LeakyReLU,
        norm_layer=nn.LayerNorm,
        sr_ratio=2,
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

class LocalAgg(nn.Module):
    def __init__(
        self,
        dim=8,
        mlp_ratio=4.0,
        drop=0.1,
        drop_path=0.1,
        act_layer=nn.LeakyReLU,

    ):
        super().__init__()
        self.pos_embed1 = nn.Conv3d(dim, dim, 3, padding=1)
        # self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn1 = nn.Conv3d(dim, dim, 5, padding=2)
        # self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
    def forward(self, x):
        x = x + self.pos_embed1(x)
        x = x + self.drop_path(self.conv2(self.attn1(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class Enhance1d(nn.Module):
    def __init__(self):
        super(Enhance1d, self).__init__()

        # Initialize modules using the provided classes
        self.enhance_conv = enhance1dconv()
        self.enhance_linear = enhance1dlinear()
        # self.enhance_transformer = Enhance1dpool()
        self.enhance_transformer=enhance1dgroupconv()
        self.tf=tf()
        self.drop=nn.Dropout(0.1)
        self.re=nn.LeakyReLU()
        self.bn = nn.BatchNorm3d(8)
        self.b1 = nn.BatchNorm3d(8)
        self.b2 = nn.BatchNorm3d(4)
        self.local=LocalAgg(dim=8)
        self.sp11=SelfAttn(dim=8,num_heads=4,sr_ratio=2)
        self.local2=LocalAgg(dim=4)
        self.sp21=SelfAttn(dim=4,num_heads=4,sr_ratio=4)
        # self.bn6 = nn.BatchNorm3d(1)
        self.sig=nn.Sigmoid()
        self.conv1 = nn.Conv3d(1, 1, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=7, stride=1, padding=3)
        self.trans_conv1 = nn.ConvTranspose3d(in_channels=8, out_channels=4,
                                             kernel_size=(4, 4, 4),
                                             stride=(2, 2, 2),
                                             padding=(1, 1, 1))
        self.trans_conv2 = nn.ConvTranspose3d(in_channels=4, out_channels=1,
                                             kernel_size=(4, 4, 4),
                                             stride=(2, 2, 2),
                                             padding=(1, 1, 1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolutional operations
        a = self.enhance_conv(x)
        b = self.enhance_linear(x)
        c = self.enhance_transformer(x)
        x=self.tf(a,b,c)
        z=x.shape[0]
        x = x.view(z, 8, 32, 32, 32)
        x=self.bn(x)
        s1=x
        x=self.local(x)
        x=self.sp11(x)
        x=s1+x
        x=self.b1(x)
        x=self.trans_conv1(x)
        s2=x
        x=self.local2(x)
        x=self.sp21(x)
        x=s2+x
        x = self.b2(x)
        x=self.trans_conv2(x)
        shortcut=x
        x=self.conv1(x)
        x=self.re(x)
        x=self.conv2(x)
        x=x+shortcut
        x=self.sig(x)
        return x

class Enhance3DSupervision(nn.Module):
    def __init__(self, in_channels: int = 16, out_channels: int = 1, kernel_size: int = 3, padding: int = 1):
        super(Enhance3DSupervision, self).__init__()
        # 初始化第一个和第二个3D自适应平均池化层
        self.pool1 = nn.AdaptiveAvgPool3d((8, 8, 8))
        self.fc1=nn.Linear(512,64)
        self.fc2=nn.Linear(64,1)
        self.drop=nn.Dropout(0.1)
        self.re=nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用两次3D自适应平均池化
        x = self.pool1(x)
        x = torch.flatten(x, start_dim=1)
        x=self.fc1(x)
        x=self.drop(x)
        x=self.re(x)
        x=self.fc2(x)
        x = self.sig(x)
        x = x * self.output_range + self.output_shift
        return x

class tf(nn.Module):
    def __init__(self):
        super(tf, self).__init__()
        self.bl1=nn.Bilinear(64, 64, 63)
        self.bl2 = nn.Bilinear(64, 64, 63)
        self.bl3 = nn.Bilinear(64, 64, 63)
        self.drop=nn.Dropout(0.1)
        self.ge=nn.LeakyReLU()
    def forward(self, x,y,z):
        o1=self.bl1(x,y)
        o2=self.bl2(x,z)
        o3=self.bl3(y,z)
        o1_additional = torch.ones(o1.shape[0], 1, device=o1.device, dtype=o1.dtype)
        o1 = torch.cat((o1, o1_additional), 1)
        o2_additional = torch.ones(o2.shape[0], 1, device=o2.device, dtype=o1.dtype)
        o2 = torch.cat((o2, o2_additional), 1)
        o3_additional = torch.ones(o3.shape[0], 1, device=o3.device, dtype=o3.dtype)
        o3 = torch.cat((o3, o3_additional), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        return o123

class enhance1dconv(nn.Module):
    def __init__(self):
        super(enhance1dconv, self).__init__()
        self.drop=nn.Dropout(0.1)
        self.conv1d1 = nn.Conv1d(127, 256, kernel_size=1, stride=1, padding=0)
        self.ge=nn.LeakyReLU()
        self.conv1d2 = nn.Conv1d(256, 64, kernel_size=1, stride=1, padding=0)
        self.bat=nn.BatchNorm1d(64)
        # self.bn=nn.BatchNorm1d(256)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 增加一个维度，将形状从[batch_size, length]变为[batch_size, channels, length]
        # 然后将通道数和长度对调，因为我们希望增加的是长度而不是通道数
        x = x.unsqueeze(-1)  # 变为[batch_size, length, 1]
        x=self.conv1d1(x)
        # x=self.bn(x)
        x=self.ge(x)
        x=self.drop(x)
        x=self.conv1d2(x)
        x = x.squeeze(-1)
        x=self.bat(x)

        return x

class enhance1dlinear(nn.Module):
    def __init__(self, in_channels: int = 16, out_channels: int = 1, kernel_size: int = 3, padding: int = 1):
        super(enhance1dlinear, self).__init__()
        self.drop=nn.Dropout(0.1)
        self.fc1 = nn.Linear(127,256)
        self.ge=nn.LeakyReLU()
        self.fc2 = nn.Linear(256,64)
        self.bat=nn.BatchNorm1d(64)
        # self.bn=nn.BatchNorm1d(256)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.fc1(x)
        # x=self.bn(x)
        x=self.drop(x)
        x=self.ge(x)
        x=self.fc2(x)
        x=self.bat(x)

        return x

class enhance1dgroupconv(nn.Module):
    def __init__(self):
        super(enhance1dgroupconv, self).__init__()
        self.drop = nn.Dropout(0.1)
        # 分组卷积层
        self.po = nn.AdaptiveAvgPool1d(128)
        self.conv1 = nn.Conv1d(128, 256, kernel_size=1, padding=0, groups=128)
        # 第二个分组卷积层
        self.conv2 = nn.Conv1d(256, 64, kernel_size=1, padding=0, groups=64)
        self.bat = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.po(x)
        x = x.unsqueeze(-1)  # 变为[batch_size, length, 1]
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x.squeeze(-1)
        x = self.bat(x)
        return x

