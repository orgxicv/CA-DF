import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import SpatialTransformer, ResNet, BottleneckBlock, Unet
from timm.models.layers import get_attn, LayerNorm2d, DropPath

def conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=False,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm2d(out_channels)
        else:
            nm = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, nm, relu)


class AggregationNetwork(nn.Module):
    """
    Module for aggregating feature maps across time and space.
    Design inspired by the Feature Extractor from ODISE (Xu et. al., CVPR 2023).
    https://github.com/NVlabs/ODISE/blob/5836c0adfcd8d7fd1f8016ff5604d4a31dd3b145/odise/modeling/backbone/feature_extractor.py
    """
    def __init__(
            self, 
            device, 
            feature_dims=[768, 768],
            projection_dim=256,
            num_norm_groups=32,
            save_timestep=[1],
            kernel_size = [1,3,1],
            contrastive_temp = 10,
            feat_map_dropout=0.0,
        ):
        super().__init__()
        self.skip_connection = True
        self.feat_map_dropout = feat_map_dropout
        self.azimuth_embedding = None
        self.pos_embedding = None
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.self_logit_scale = nn.Parameter(torch.ones([]) * np.log(contrastive_temp))
        self.device = device
        self.save_timestep = save_timestep
        
        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=1,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups,
                    kernel_size=kernel_size
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
            for t in save_timestep:
                # 1-index the layer name following prior work
                self.mixing_weights_names.append(f"timestep-{save_timestep}_layer-{l+1}")
        self.last_layer = None
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))
        # count number of parameters
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(f"AggregationNetwork has {num_params} parameters.")
    
    def load_pretrained_weights(self, pretrained_dict):
        custom_dict = self.state_dict()

        # Handle size mismatch
        if 'mixing_weights' in custom_dict and 'mixing_weights' in pretrained_dict and custom_dict['mixing_weights'].shape != pretrained_dict['mixing_weights'].shape:
            # Keep the first four weights from the pretrained model, and randomly initialize the fifth weight
            custom_dict['mixing_weights'][:4] = pretrained_dict['mixing_weights'][:4]
            custom_dict['mixing_weights'][4] = torch.zeros_like(custom_dict['mixing_weights'][4])
        else:
            custom_dict['mixing_weights'][:4] = pretrained_dict['mixing_weights'][:4]

        # Load the weights that do match
        matching_keys = {k: v for k, v in pretrained_dict.items() if k in custom_dict and k != 'mixing_weights'}
        custom_dict.update(matching_keys)

        # Now load the updated state_dict
        self.load_state_dict(custom_dict, strict=False)
        
    def forward(self, batch, pose=None):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        if self.feat_map_dropout > 0 and self.training:
            batch = F.dropout(batch, p=self.feat_map_dropout)
        
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights, dim=0)
        if self.pos_embedding is not None: #position embedding
            batch = torch.cat((batch, self.pos_embedding), dim=1)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        
        if self.last_layer is not None:

            output_feature_after = self.last_layer(output_feature)
            if self.skip_connection:
                # skip connection
                output_feature = output_feature + output_feature_after
        return output_feature
    

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            # x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            # x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class MbConvBlock(nn.Module):
    """ A depthwise separable / fused mbconv style residual block with SE, `no norm.
    """
    def __init__(
            self,
            in_chs,
            out_chs=None,
            expand_ratio=1.0,
            attn_layer='se',
            bias=False,
            act_layer=nn.GELU,
            norm_layer=LayerNorm2d,
            drop_path_rate=0.0,
    ):
        super().__init__()
        attn_kwargs = dict(act_layer=act_layer)
        if isinstance(attn_layer, str) and attn_layer == 'se' or attn_layer == 'eca':
            attn_kwargs['rd_ratio'] = 0.25
            attn_kwargs['bias'] = False
        attn_layer = get_attn(attn_layer)
        out_chs = out_chs or in_chs
        mid_chs = int(expand_ratio * in_chs)

        self.pre_norm = norm_layer(in_chs) if norm_layer is not None else nn.Identity()
        self.conv_dw = nn.Conv2d(in_chs, mid_chs, 3, 1, 1, groups=in_chs, bias=bias)
        self.act = act_layer()
        self.se = attn_layer(mid_chs, **attn_kwargs)
        self.conv_pw = nn.Conv2d(mid_chs, out_chs, 1, 1, 0, bias=bias)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.pre_norm(x)
        x = self.conv_dw(x)
        x = self.act(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.drop_path(x) + shortcut
        return x

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CAA(nn.Module):
    """Channel Aggregation and Attention module"""
    def __init__(self, projection_dim=256):
        super(CAA, self).__init__()
        self.aggrenet = AggregationNetwork(device='cuda', projection_dim=projection_dim, feat_map_dropout=0.2)
        self.chs_attn = MbConvBlock(projection_dim, projection_dim)
        
    def forward(self, x):
        x = self.aggrenet(x)
        x = self.chs_attn(x)
        return x

class DispDecoder(nn.Module):
    def __init__(self, inshape=[192,192]):
        super().__init__()

        self.flow_refine1 = nn.Conv2d(512, 2, kernel_size=3, padding=1)
        self.spa1 = SpatialTransformer(volsize=[inshape[0] // 4, inshape[1] // 4])

        self.upsample1 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.conv2 = conv2D(64, 32, kernel_size=3, padding=1)
        self.flow_refine2 = nn.Conv2d(66, 2, kernel_size=3, padding=1)
        self.spa2 = SpatialTransformer(volsize=[inshape[0] // 2, inshape[1] // 2])

        self.upsample2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = conv2D(16, 8, kernel_size=3, padding=1)
        self.flow_refine3 = nn.Conv2d(18, 2, kernel_size=3, padding=1)

        self.dc_conv0 = conv2D(18, 12, kernel_size=3, padding=1)
        self.dc_conv1 = conv2D(12, 8, kernel_size=3, padding=1)
        self.dc_conv2 = conv2D(8, 4, kernel_size=3, padding=1)
        self.predict_flow = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2)

        self.spa3 = SpatialTransformer(volsize=[inshape[0], inshape[1]])

        self.resize1 = ResizeTransform(1 / 2, 2)
        self.resize2 = ResizeTransform(1 / 2, 2)
        self.resize = ResizeTransform(1 / 4, 2)

    def forward(self, fix_caa, moving_caa):
        flow1 = self.flow_refine1(torch.cat([moving_caa, fix_caa], dim=1))
        
        # First upsampling level
        x_up1 = self.upsample1(moving_caa)
        y_up1 = self.upsample1(fix_caa)
        x_up1 = self.conv2(x_up1)
        y_up1 = self.conv2(y_up1)
        flow2 = self.flow_refine2(torch.cat([x_up1, y_up1, self.resize1(flow1)], dim=1))
        
        # Second upsampling level
        x_up2 = self.upsample2(x_up1)
        y_up2 = self.upsample2(y_up1)
        x_up2 = self.conv3(x_up2)
        y_up2 = self.conv3(y_up2)
        flow3 = self.flow_refine3(torch.cat([x_up2, y_up2, self.resize2(flow2)], dim=1))
        
        # Final flow refinement
        flow_field = self.dc_conv0(torch.cat([x_up2, y_up2, flow3], dim=1))
        flow_field = self.dc_conv1(flow_field)
        flow_field = self.dc_conv2(flow_field)
        flow_field = self.relu(self.predict_flow(flow_field)) + flow3
        
        return flow_field

class RegisNet(nn.Module):
    def __init__(self, inshape=[192,192], projection_dim=256):
        super(RegisNet, self).__init__()
        self.caa = CAA(projection_dim=projection_dim)
        self.disp_decoder = DispDecoder(inshape=inshape)
        self.resize = ResizeTransform(1 / 4, 2)
        self.spa3 = SpatialTransformer(volsize=[inshape[0], inshape[1]])
        self.similarity = nn.L1Loss()

    def forward(self, fix_feats, moving_feats):
        # Process features through CAA
        fix_caa = self.caa(fix_feats)
        moving_caa = self.caa(moving_feats)
        
        # Get displacement field
        flow_field = self.disp_decoder(fix_caa, moving_caa)
        
        # Transform moving image
        moving = self.resize(torch.cat([moving_feats,moving_caa], dim=1))
        fix = self.resize(torch.cat([fix_feats,fix_caa], dim=1))
        y_pred = self.spa3(moving, flow_field)
        
        # Currently not using haloss
        tmploss = 0
        return flow_field, y_pred, fix, tmploss
