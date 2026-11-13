
# ============================================================
# BLOCK 3: MODIFIED CBAM
# ------------------------------------------------------------
# IMPROVEMENTS OVER STANDARD CBAM:
#
# 1. Added Standard Deviation Pooling
# 2. Dual-path Channel Attention
# 3. Concatenation-based Fusion
# 4. Enhanced Spatial Attention
#
# IMPORTANT:
# Added sigmoid stabilization before feature scaling
# to prevent unstable attention magnitudes.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# STANDARD DEVIATION POOLING
# ============================================================

class StdPool(nn.Module):

    def forward(self, x):

        return torch.std(x, dim=(2, 3), keepdim=True)


# ============================================================
# SPATIAL STD POOLING
# ============================================================

class SpatialStdPool(nn.Module):

    def forward(self, x):

        return torch.std(x, dim=1, keepdim=True)


# ============================================================
# MODIFIED CHANNEL ATTENTION
# ============================================================

class DualPathConcatChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):

        super().__init__()

        # ----------------------------------------------------
        # Pooling layers
        # ----------------------------------------------------

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.std_pool = StdPool()

        # ----------------------------------------------------
        # Shared MLP
        # ----------------------------------------------------

        self.mlp = nn.Sequential(

            nn.Linear(
                in_channels,
                in_channels // reduction,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                in_channels // reduction,
                in_channels,
                bias=False
            )
        )

        # ----------------------------------------------------
        # Activations
        # ----------------------------------------------------

        self.relu = nn.ReLU(inplace=True)

        self.gelu = nn.GELU()

        # ----------------------------------------------------
        # Fusion Conv
        # Compress:
        # 2C → C
        # ----------------------------------------------------

        self.channel_fusion = nn.Conv2d(
            in_channels * 2,
            in_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):

        B, C, H, W = x.shape

        # ====================================================
        # UPPER PATH
        # Avg + Max + ReLU
        # ====================================================

        max_out = self.max_pool(x).view(B, C)

        avg_out_upper = self.avg_pool(x).view(B, C)

        max_out = self.mlp(max_out)

        avg_out_upper = self.mlp(avg_out_upper)

        upper_attention = max_out + avg_out_upper

        upper_attention = self.relu(upper_attention)

        upper_attention = upper_attention.view(B, C, 1, 1)

        # ====================================================
        # LOWER PATH
        # Avg + Std + GELU
        # ====================================================

        avg_out_lower = self.avg_pool(x).view(B, C)

        std_out = self.std_pool(x).view(B, C)

        avg_out_lower = self.mlp(avg_out_lower)

        std_out = self.mlp(std_out)

        lower_attention = avg_out_lower + std_out

        lower_attention = self.gelu(lower_attention)

        lower_attention = lower_attention.view(B, C, 1, 1)

        # ====================================================
        # CONCATENATION
        # ====================================================

        concat_attention = torch.cat(
            [upper_attention, lower_attention],
            dim=1
        )

        # ====================================================
        # CHANNEL FUSION
        # 2C → C
        # ====================================================

        fused_attention = self.channel_fusion(concat_attention)

        # ====================================================
        # IMPORTANT STABILIZATION
        # ----------------------------------------------------
        # Prevents exploding feature scaling
        # ====================================================

        fused_attention = torch.sigmoid(fused_attention)

        # ====================================================
        # APPLY CHANNEL ATTENTION
        # ====================================================

        out = x * fused_attention

        return out


# ============================================================
# MODIFIED SPATIAL ATTENTION
# ============================================================

class ModifiedSpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super().__init__()

        self.std_pool = SpatialStdPool()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # ----------------------------------------------------
        # Avg Pool
        # ----------------------------------------------------

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # Max Pool
        # ----------------------------------------------------

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # Std Pool
        # ----------------------------------------------------

        std_out = self.std_pool(x)

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        concat = torch.cat(
            [avg_out, max_out, std_out],
            dim=1
        )

        # ----------------------------------------------------
        # Spatial attention
        # ----------------------------------------------------

        attention = self.conv(concat)

        attention = self.sigmoid(attention)

        # ----------------------------------------------------
        # Apply attention
        # ----------------------------------------------------

        out = x * attention

        return out


# ============================================================
# COMPLETE MODIFIED CBAM
# ============================================================

class ModifiedCBAM(nn.Module):

    def __init__(
        self,
        in_channels,
        reduction=16,
        kernel_size=7
    ):

        super().__init__()

        # ----------------------------------------------------
        # Channel Attention
        # ----------------------------------------------------

        self.channel_attention = (
            DualPathConcatChannelAttention(
                in_channels,
                reduction
            )
        )

        # ----------------------------------------------------
        # Spatial Attention
        # ----------------------------------------------------

        self.spatial_attention = (
            ModifiedSpatialAttention(
                kernel_size
            )
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Channel Attention
        # ----------------------------------------------------

        x = self.channel_attention(x)

        # ----------------------------------------------------
        # Spatial Attention
        # ----------------------------------------------------

        x = self.spatial_attention(x)

        return x
