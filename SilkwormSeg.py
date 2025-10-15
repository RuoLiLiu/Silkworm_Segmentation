import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_, DropPath, SqueezeExcite
import math
from mmcv.cnn.bricks import Swish
from mmengine.model import BaseModule
from typing import List, Dict, Union

from .kan import KANLinear


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
    
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        # x = self.fc3(x.reshape(B*N,C))
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_3(x, H, W)
    
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = Conv2d_BN(dim, dim, 3, 1, 1, groups=dim)
        self.relu = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Conv2d_BN(torch.nn.Sequential): 
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):

        c, bn = self._modules.values()

        w = bn.weight / (bn.running_var + bn.eps)**0.5 
        w = c.weight * w[:, None, None, None]  
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5 

        m = torch.nn.Conv2d(w.size(1) * self.c.groups,
                            w.size(0),                
                            w.shape[2:],               
                            stride=self.c.stride, 
                            padding=self.c.padding, 
                            dilation=self.c.dilation, 
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # global max pooling

        # Pointwise convolution, reducing the number of channels to in_planes//16
        self.fc1 = Conv2d_BN(in_planes, in_planes // ratio, 1, 1, 0)
        self.relu = nn.GELU()
        # Pointwise convolution, restoring the number of channels to in_planes
        self.fc2 = Conv2d_BN(in_planes // ratio, in_planes, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = Conv2d_BN(2, 1, ks=kernel_size, stride=1, pad=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class RepCAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=3):
        super(RepCAM, self).__init__()

        self.rep_ca = ChannelAttention(in_planes, ratio)
        self.rep_sa = SpatialAttention(kernel_size)

    def forward(self, x):

        out = self.rep_ca(x) * x
        out = self.rep_sa(out) * out

        return out

class Residual(torch.nn.Module): 
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:

            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self

class InvertedResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.ib = Residual(nn.Sequential(
            Conv2d_BN(dim, dim, 3, 1, 1, groups=dim),
            Conv2d_BN(dim, 2 * dim, 1, 1, 0),
            nn.GELU(),
            Conv2d_BN(2 * dim, dim, 1, 1, 0, bn_weight_init=0)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.ib(x)

        return x

class PathStage(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 first_time: bool = False,
                 last_time: bool = False,
                 conv_bn_act_pattern: bool = False,
                 epsilon: float = 1e-4) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_time = first_time
        self.last_time = last_time
        self.conv_bn_act_pattern = conv_bn_act_pattern 
        self.epsilon = epsilon

        if self.first_time:
            self.p5_down_channel = Conv2d_BN(self.in_channels[-1], self.out_channels, 1, 1, 0)
            self.p4_down_channel = Conv2d_BN(self.in_channels[-2], self.out_channels, 1, 1, 0)
            self.p3_down_channel = Conv2d_BN(self.in_channels[-3], self.out_channels, 1, 1, 0)

            self.p4_level_connection = Conv2d_BN(self.in_channels[-2], self.out_channels, 1, 1, 0)

        ### bottom to up: feature map down_sample module
        self.p4_down_sample = Conv2d_BN(self.in_channels[-1], self.in_channels[-1], 3, 2, 1, groups=self.in_channels[-1])
        self.p5_down_sample = Conv2d_BN(self.in_channels[-1], self.in_channels[-1], 3, 2, 1, groups=self.in_channels[-1])

        ### Up-sampling
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')  ### P5_in to P4_up
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')  ### P4_in to P3_up

        # Fuse Conv Layers
        self.conv4_up = nn.Sequential(
            Conv2d_BN(self.out_channels, self.out_channels, 3, 1, 1, groups=self.out_channels),
            Conv2d_BN(self.out_channels, self.out_channels, 1, 1, 0))
        self.conv3_up = nn.Sequential(
            Conv2d_BN(self.out_channels, self.out_channels, 3, 1, 1, groups=self.out_channels),
            Conv2d_BN(self.out_channels, self.out_channels, 1, 1, 0))

        
        self.conv4_down = nn.Sequential(
            Conv2d_BN(self.out_channels, self.out_channels, 3, 1, 1, groups=self.out_channels),
            Conv2d_BN(self.out_channels, self.out_channels, 1, 1, 0))
        self.conv5_down = nn.Sequential(
            Conv2d_BN(self.out_channels, self.out_channels, 3, 1, 1, groups=self.out_channels),
            Conv2d_BN(self.out_channels, self.out_channels, 1, 1, 0))


        self.conv3_out = Conv2d_BN(out_channels, self.in_channels[-3], 1, 1, 0)
        self.conv4_out = Conv2d_BN(out_channels, self.in_channels[-2], 1, 1, 0)
        self.conv5_out = Conv2d_BN(out_channels, self.in_channels[-1], 1, 1, 0)

        # weights
        self.p4_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.GELU()
        self.p3_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.GELU()


        self.p4_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.GELU()
        self.p5_w2 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.GELU()

        self.swish = Swish()

    def combine(self, x):
        if not self.conv_bn_act_pattern:
            x = self.swish(x)

        return x

    def forward(self, x):
        if self.first_time:  
            p3, p4, p5 = x  

            p3_in = self.p3_down_channel(p3)  
            p4_in = self.p4_down_channel(p4) 
            p5_in = self.p5_down_channel(p5) 

        else: 
            p3_in, p4_in, p5_in = x


        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(
            self.combine(weight[0] * p4_in +
                         weight[1] * self.p4_upsample(p5_in)))

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(
            self.combine(weight[0] * p3_in +
                         weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_level_connection(p4)


        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            self.combine(weight[0] * p4_in + weight[1] * p4_up +
                         weight[2] * self.p4_down_sample(p3_out)))

        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(
            self.combine(weight[0] * p5_in + 
                         weight[1] * self.p5_down_sample(p4_out)))

        if self.last_time:
            p3_out = self.conv3_out(p3_out)
            p4_out = self.conv4_out(p4_out)
            p5_out = self.conv5_out(p5_out)

        return p3_out, p4_out, p5_out

class ResPath(BaseModule):
    def __init__(self,
                 num_stages: int,
                 in_channels: List[int],
                 out_channels: int,
                 start_level: int = 0,
                 epsilon: float = 1e-4,
                 conv_bn_act_pattern: bool = False,
                 init_cfg: Union[Dict, List[Dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.start_level = start_level
        self.res_path = nn.Sequential(*[
            PathStage(
                in_channels=in_channels,
                out_channels=out_channels,
                first_time=True if _ == 0 else False,
                last_time = True if _ == num_stages - 1 else False,
                conv_bn_act_pattern=conv_bn_act_pattern,
                epsilon=epsilon) for _ in range(num_stages)
        ])

    def forward(self, x):
        x = x[self.start_level:]
        x = self.res_path(x)

        return x


class SilkwormSeg(nn.Module):
    def __init__(self, num_classes=1, img_size=512, embed_dims=[16, 32, 64, 128], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        ### Encoder
        self.encoder_layers = nn.ModuleDict()
        current_depth = 0
        stage = 0
        for i in range(sum(depths)):
            if stage not in self.encoder_layers:
                self.encoder_layers[str(stage)] = nn.ModuleList()
            if i == 0 or i == sum(depths[:stage]):  # stage 0, 1, 2
                self.encoder_layers[str(stage)].append(nn.Sequential(
                    Conv2d_BN(3 if stage == 0 else embed_dims[stage-1], embed_dims[stage], 3, 2, 1, 
                               groups=embed_dims[stage-1] if stage > 0 else 1),
                    nn.GELU(),
                    RepCAM(embed_dims[stage]),
                    InvertedResidualBlock(dim=embed_dims[stage]),
                ))
            else:
                self.encoder_layers[str(stage)].append(nn.Sequential(
                    Conv2d_BN(embed_dims[stage], embed_dims[stage], 3, 1, 1, groups=embed_dims[stage]),
                    nn.GELU(),
                    RepCAM(embed_dims[stage]),
                    InvertedResidualBlock(dim=embed_dims[stage]),
                ))
            current_depth += 1
            if current_depth == depths[stage]:
                stage += 1
                current_depth = 0
                
        
        ### Bottleneck
        self.patch_embed = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.bottleneck = nn.ModuleList([KANBlock(
            dim=embed_dims[3],   # 256
            drop=drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,  # 0.0 0.0
            no_kan=no_kan
            )])
        self.norm = norm_layer(embed_dims[3])

        ### Decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Sequential(
                    Conv2d_BN(embed_dims[3], embed_dims[2], 3, 1, 1),
                    nn.GELU(),
                    RepCAM(embed_dims[2]),
                    InvertedResidualBlock(dim=embed_dims[2]),
                ))
        self.decoder_layers.append(nn.Sequential(
                    Conv2d_BN(embed_dims[2], embed_dims[1], 3, 1, 1),
                    nn.GELU(),
                    RepCAM(embed_dims[1]),
                    InvertedResidualBlock(dim=embed_dims[1]),
                ))
        self.decoder_layers.append(nn.Sequential(
                    Conv2d_BN(embed_dims[1], embed_dims[0], 3, 1, 1),
                    nn.GELU(),
                    RepCAM(embed_dims[0]),
                    InvertedResidualBlock(dim=embed_dims[0]),
                ))
        self.decoder_layers.append(nn.Sequential(
                    Conv2d_BN(embed_dims[0], embed_dims[0], 3, 1, 1),
                    nn.GELU(),
                    RepCAM(embed_dims[0]),
                    InvertedResidualBlock(dim=embed_dims[0]),
                ))

        ### Skip connections
        self.path = ResPath(num_stages=2, in_channels=embed_dims[:-1], out_channels=embed_dims[-2])
        
        self.upconv0 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[3], embed_dims[2], kernel_size=2, stride=2),  
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU()
        )
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[2], embed_dims[1], kernel_size=2, stride=2),  
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU()
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[1], embed_dims[0], kernel_size=2, stride=2),  
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[0], embed_dims[0], kernel_size=2, stride=2),  
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )


        self.final = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1) 


    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        features = []
        for stage in self.encoder_layers:
            for layer in self.encoder_layers[stage]:
                x = layer(x)
                features.append(x)
        
        t = self.path(features)

        ### Bottleneck
        out, H, W = self.patch_embed(features[2])
        for i, kan_layer in enumerate(self.bottleneck):
            out = kan_layer(out, H, W)
        out = self.norm(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Decoder
        out = self.upconv0(out)
        out = torch.cat([out, t[2]], dim=1)
        out = self.decoder_layers[0](out)
        
        out = self.upconv1(out)
        out = torch.cat([out, t[1]], dim=1)
        out = self.decoder_layers[1](out)
        
        out = self.upconv2(out)
        out = torch.cat([out, t[0]], dim=1)
        out = self.decoder_layers[2](out)

        
        ### SegHead
        out = self.upconv3(out)
        out = self.final(out)

        return out





# if __name__ == '__main__':

#     model = SilkwormSeg(num_classes=1).cuda()

#     x = torch.randn(1, 3, 256, 256).cuda()
#     model.eval()
#     y = model(x)
#     print(y.shape)



