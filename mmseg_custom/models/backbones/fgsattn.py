import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class ChannelPool(nn.Module):
    def __init__(self, pool_types=['avg', 'max']):
        super(ChannelPool, self).__init__()
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = torch.mean(x,1).unsqueeze(1)
                channel_att_raw = avg_pool
            elif pool_type=='max':
                max_pool = torch.max(x,1)[0].unsqueeze(1)
                channel_att_raw = max_pool

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        return channel_att_sum
    
class FGSAttn(nn.Module):
    def __init__(self, dim, double_R, group=1, d=1, pool_types=['avg', 'max'], init_values=0.):
        super(FGSAttn, self).__init__()
        self.compress = ChannelPool(pool_types)
        self.group = group
        self.fre_interval = d
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        channel = int(double_R/2/d)+1
        self.fc = nn.Sequential(*[nn.Sequential(
            nn.Linear(channel, channel, bias=True),
            nn.LeakyReLU()
        )for _ in range(group)])          # unshare
        
    def min_max(self, x, to_min, to_max):
        x_min = torch.min(x)
        x_max = torch.max(x)
        return to_min + ((to_max - to_min) / (x_max - x_min)) * (x - x_min)
        
    def forward(self, feature):
        B, C, H, W = feature.shape
        group = self.group
        K = int(C/group)
        feature1 = None
        for i in range(group):

            feature_i = feature[:, i*K:(i+1)*K, :, :]                             # BKHW

            feature_compress_i = self.compress(feature_i).squeeze(1)              # B1HW -> BHW
            new_feature = []
            for b in range(B):
                feature_map = feature_compress_i[b]
                f = torch.fft.fft2(feature_map)
                shift2center = torch.fft.fftshift(f)
                amplitude_spectrum = torch.abs(shift2center)              # 振幅谱
                phase_spectrum = torch.angle(shift2center)                # 相位谱    
                H1, W1 = amplitude_spectrum.shape
                center_h = int(H1/2)
                center_w = int(W1/2)
                R = min(center_h, center_w)
                d = self.fre_interval     
                mask = torch.zeros((H1, W1), dtype=torch.int64)
                for h in range(H1):
                    for w in range(W1):
                        h1 = h - center_h
                        w1 = w - center_w
                        r = np.sqrt(h1*h1+w1*w1)
                        r_floor = np.floor(r)
                        if r_floor > R:
                            r_floor = R
                        label = int(np.ceil(r_floor/d))
                        mask[h, w] = label
                
                mask_Label = mask.unique()
                assert(len(mask_Label)==int(R/d)+1)     
                fre_avg_pool = []
                for label in mask_Label:
                    fre_avg_pool.append(torch.mean(amplitude_spectrum[mask==label]))
                fre_avg_pool = torch.tensor(fre_avg_pool)
                fre_att = self.fc[i](fre_avg_pool.to(feature.device))
                amplitude_spectrum1 = amplitude_spectrum.clone()
                for index in range(len(fre_att)):
                    label = index
                    attn = fre_att[index]
                    amplitude_spectrum1[mask==label]=(amplitude_spectrum*attn)[mask==label]
                                
                merged_feature = torch.multiply(torch.exp(1j * phase_spectrum), amplitude_spectrum1)
                out = torch.fft.ifftshift(merged_feature)
                out1 = torch.fft.ifft2(out)
                new_feature_map = torch.real(out1)
                attention_map = self.min_max(new_feature_map, 0, 1)    
                out_feature_i = feature_i[b] * attention_map                                       # KHW
                new_feature.append(out_feature_i)
                            
            new_feature = torch.stack(new_feature, dim=0).unsqueeze(1)         # BKHW
            if feature1 is None:
                feature1 = new_feature
            else:
                feature1 = torch.cat([feature1, new_feature], dim=1)
        feature = feature.view(B, C, -1).transpose(1, 2) 
        feature1 = feature1.view(B, C, -1).transpose(1, 2) 
        out_feature = feature + self.gamma * feature1
        out_feature = out_feature.transpose(1, 2).view(B, C, H1, W1).contiguous()
        return out_feature
                        
                        
            

