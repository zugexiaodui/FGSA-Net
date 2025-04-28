# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from .base.uniperceiver import UnifiedBertEncoder
from .fgsa_net_modules import FBNM_Module, FBFE_Module, deform_inputs

_logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class FGSA_Net(UnifiedBertEncoder):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, adaptation_indexes=None, pretrained=None, with_cp=False, group=1, d=1, pool_types=['avg'],
                 *args, **kwargs):

        super().__init__(num_heads=num_heads, pretrained=pretrained, with_cp=with_cp, *args, **kwargs)

        self.cls_token = None
        self.num_block = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.adaptation_indexes = adaptation_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim
        
        self.fbnm = FBNM_Module(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=with_cp, group=group, d=d, pool_types=pool_types,
                num_heads=deform_num_heads, n_points=n_points, norm_layer=self.norm_layer, init_values=init_values, deform_ratio=deform_ratio)
        self.adaptations = nn.Sequential(*[
            FBFE_Module(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                        init_values=init_values, drop_path=self.drop_path_rate,
                        norm_layer=self.norm_layer, with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                        extra_extractor=True if i == len(adaptation_indexes) - 1 else False, with_cp=with_cp, index=i, group=group, d=d, pool_types=pool_types)
            for i in range(len(adaptation_indexes))
        ])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)
        
        self.up.apply(self._init_weights)
        self.fbnm.apply(self._init_weights)
        self.adaptations.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups

            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        ori_x = x     # img
        # Patch Embedding forward
        x, H, W = self.visual_embed(x)
        bs, n, dim = x.shape
        
        # FBNM Forward
        c1, c2, c3, c4, x = self.fbnm(ori_x, x, deform_inputs1)
        c = torch.cat([c2, c3, c4], dim=1)
        
        # Adaptation
        outs = list()
        for i, fbfe in enumerate(self.adaptations):
            indexes = self.adaptation_indexes[i]
            for idx, blk in enumerate(self.layers[indexes[0]:indexes[-1] + 1]):  
                x = blk(x, H, W)   
            x, c, out_x = fbfe(x, c, deform_inputs1, deform_inputs2, H, W, i)
            outs.append(out_x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
