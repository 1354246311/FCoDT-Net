
from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collections.abc import Sequence
from functools import partial
from typing import Optional, Sequence, Tuple, Type, Union
import edt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, look_up_option


import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer


class FCoDT(nn.Module):

    def __init__(self, dim=512,kernel_size=3,res=False):
        super().__init__()
        self.res = res
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv3d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv3d(dim,dim,1,bias=False),
            nn.BatchNorm3d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv3d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm3d(2*dim//factor),
            nn.ReLU(),
            nn.Conv3d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, feature1,feature2):
        bs,c,h,w,d=feature1.shape
        k1=self.key_embed(feature1) #bs,c,h,w    1????
        v=self.value_embed(feature2).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,feature1],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w,d)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w,d)

        out = k1+k2

        return out


class FDB(nn.Module):
    """
    Feature Distillation Block
    """
    def __init__(self, dim, img_size,kernel_size=3):
        super(FDB, self).__init__()
        self.ln1 = nn.LayerNorm(img_size)
        self.att1 = FCoDT(dim, kernel_size)
        self.ln2 = nn.LayerNorm(img_size)
        self.att2 = FCoDT(dim, kernel_size)
        self.ln3= nn.LayerNorm(img_size)


    def forward(self, feature1, feature2):
        # 将五维特征图展平成三维

        #return  feature1+feature2

        org=feature2
        out1 = self.ln1(feature2)
        out1 = self.att1(out1,feature2) # 使用feature1作为q,k,v
        out1=out1+feature2

        out1 = self.ln2(out1)

        feature1=self.ln2(feature1)
        out2 = self.att2(feature1, out1)  # 使用feature1作为q，feature2作为k,v

        out2=out2+out1+org

        out=self.ln3(out2)
        #out=self.att3(out2,out2)
        #out=out+org

        # 将三维特征图恢复成五维
        return out



class DDB(nn.Module):
    """
    Decoder Distillation Block
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        img_size:int,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        self.FDB=FDB(out_channels, img_size=img_size)
    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        skip=self.FDB(out,skip)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out




class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            x = x.permute(0, 2, 3, 4, 1)
            x=F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x=x.permute(0, 4, 1, 2, 3)
            return x
        elif self.data_format == "channels_first":
            x = x.permute(1, 2, 3, 4, 0)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(4, 0, 1, 2, 3)
            return x
            #x=x.permute(1, 0, 2, 3, 4)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None, None] * x + self.bias[:, None, None, None, None]
            #weight = self.weight[:, None, None, None, None].contiguous()
            #bias = self.bias[:, None, None, None, None].contiguous()
            #x = weight * x + bias

            #x=x.permute(1, 0, 2, 3, 4)
            return x

class Block(nn.Module):



    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        bs,c,w,h,d=x.size()
        x = self.dwconv(x)
        x = x.permute(1, 0, 2, 3, 4)  # (N, C, H, W, D) -> (C, N, H, W, D)
        x = self.norm(x)
        x=x.reshape(-1,c)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x=x.view(c,bs,w,h,d)
        x = x.permute(1, 0, 2, 3, 4)  # (C, N, H, W, D) -> (N, C, H, W, D)
        x = input + x
        #x = input + self.drop_path(x)
        return x


class convEncoder(nn.Module):

    def __init__(self, in_chans=3, depths=[3 ,3, 3, 9, 3], dims=[24, 48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3,4],
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(4):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(5):
            stage = nn.Sequential(
                *[Block(dim=dims[i],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6)
        for i_layer in range(5):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        outs = []
        for i in range(5):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x




class FCoDT_NET(nn.Module):

  def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.5,
        attn_drop_rate: float = 0.5,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
    ) -> None:
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize


        self.encoder1 = UnetrBasicBlock(
          spatial_dims=spatial_dims,
          in_channels=in_channels,
          out_channels=feature_size,
          kernel_size=3,
          stride=1,
          norm_name=norm_name,
          res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
          spatial_dims=spatial_dims,
          in_channels=feature_size,
          out_channels=feature_size,
          kernel_size=3,
          stride=1,
          norm_name=norm_name,
          res_block=True,
            )

        self.encoder3 = UnetrBasicBlock(
          spatial_dims=spatial_dims,
          in_channels=2 * feature_size,
          out_channels=2 * feature_size,
          kernel_size=3,
          stride=1,
          norm_name=norm_name,
          res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
          spatial_dims=spatial_dims,
          in_channels=4 * feature_size,
          out_channels=4 * feature_size,
          kernel_size=3,
          stride=1,
          norm_name=norm_name,
          res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
          spatial_dims=spatial_dims,
          in_channels=16 * feature_size,
          out_channels=16 * feature_size,
          kernel_size=3,
          stride=1,
          norm_name=norm_name,
          res_block=True,
        )

        self.decoder5 = DDB(
          spatial_dims=spatial_dims,
          in_channels=16 * feature_size,
          out_channels=8 * feature_size,
          kernel_size=3,
          upsample_kernel_size=2,
          norm_name=norm_name,
          img_size=int(img_size[0]/16),
          res_block=True,
        )

        self.decoder4 = DDB(
          spatial_dims=spatial_dims,
          in_channels=feature_size * 8,
          out_channels=feature_size * 4,
          kernel_size=3,
          upsample_kernel_size=2,
          norm_name=norm_name,
          img_size=int(img_size[0]/8),
          res_block=True,
        )

        self.decoder3 = DDB(
          spatial_dims=spatial_dims,
          in_channels=feature_size * 4,
          out_channels=feature_size * 2,
          kernel_size=3,
          upsample_kernel_size=2,
          norm_name=norm_name,
          img_size=int(img_size[0] / 4),
          res_block=True,
        )
        self.decoder2 = DDB(
          spatial_dims=spatial_dims,
          in_channels=feature_size * 2,
          out_channels=feature_size,
          kernel_size=3,
          upsample_kernel_size=2,

          norm_name=norm_name,
          img_size=int(img_size[0] / 2),
          res_block=True,
        )

        self.decoder1 = DDB(
          spatial_dims=spatial_dims,
          in_channels=feature_size,
          out_channels=feature_size,
          kernel_size=3,
          upsample_kernel_size=2,
          norm_name=norm_name,
          img_size=int(img_size[0]),
          res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.encoders=convEncoder(in_channels)



  def forward(self, x_in):

    hidden_states_out = self.encoders(x_in)
    enc0 = self.encoder1(x_in)

    enc1 = self.encoder2(hidden_states_out[0])



    enc2 = self.encoder3(hidden_states_out[1])



    enc3 = self.encoder4(hidden_states_out[2])


    dec4 = self.encoder10(hidden_states_out[4])


    dec3 = self.decoder5(dec4, hidden_states_out[3])

    dec2 = self.decoder4(dec3, enc3)


    dec1 = self.decoder3(dec2, enc2)

    dec0 = self.decoder2(dec1, enc1)

    out = self.decoder1(dec0, enc0)

    logits = self.out(out)

    return logits
