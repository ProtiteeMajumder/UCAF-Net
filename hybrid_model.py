import torch
import torch.nn as nn
from resnet50_cbam import ResNet50CBAM
from mobilenetv2_cbam import MobileNetCBAM
from fusion_modules import FeatureAlignment, CrossAttention, UncertaintyHead, GatedFusion

# ============================================================
# BLOCK 10: FINAL HYBRID MODEL
# ------------------------------------------------------------
# UPDATES:
# 1. Attention Pooling
# 2. Uncertainty-Guided Fusion
# ============================================================

class HybridModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # ====================================================
        # BACKBONES
        # ====================================================

        self.resnet = ResNet50CBAM()

        self.mobilenet = MobileNetCBAM()

        # ====================================================
        # FEATURE ALIGNMENT
        # ====================================================

        self.align = FeatureAlignment()

        # ====================================================
        # CROSS ATTENTION
        # ====================================================

        self.cross_attn = CrossAttention(512)

        # ====================================================
        # UNCERTAINTY HEADS
        # ====================================================

        self.uncertainty_r = UncertaintyHead(512)

        self.uncertainty_m = UncertaintyHead(512)

        # ====================================================
        # UNCERTAINTY-GUIDED FUSION
        # ====================================================

        self.fusion = GatedFusion(512)

        # ====================================================
        # ATTENTION POOLING
        # ----------------------------------------------------
        # Learns which tokens are important
        # ====================================================

        self.attention_pool = nn.Sequential(

            nn.Linear(512, 256),

            nn.Tanh(),

            nn.Linear(256, 1)
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(512, 256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # ====================================================
        # BACKBONE FEATURES
        # ====================================================

        fr = self.resnet(x)      # (B,2048,H,W)

        fm = self.mobilenet(x)   # (B,1280,H,W)

        # ====================================================
        # CHANNEL ALIGNMENT
        # ====================================================

        fr, fm = self.align(fr, fm)

        # ====================================================
        # TOKENIZATION
        # ====================================================

        B, C, H, W = fr.shape

        fr = fr.view(B, C, -1).permute(0, 2, 1)

        fm = fm.view(B, C, -1).permute(0, 2, 1)

        # ====================================================
        # UNCERTAINTY ESTIMATION
        # ====================================================

        sigma_r = self.uncertainty_r(fr)

        sigma_m = self.uncertainty_m(fm)

        # ====================================================
        # BIDIRECTIONAL CROSS ATTENTION
        # ====================================================

        fr_ = self.cross_attn(fr, fm, sigma_m)

        fm_ = self.cross_attn(fm, fr, sigma_r)

        # ====================================================
        # UNCERTAINTY-GUIDED FUSION
        # ====================================================

        fused_tokens = self.fusion(
            fr_,
            fm_,
            sigma_r,
            sigma_m
        )

        # ====================================================
        # ATTENTION POOLING
        # ----------------------------------------------------
        # Learns important diagnostic regions
        # ====================================================

        attention_scores = self.attention_pool(fused_tokens)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        fused = torch.sum(
            attention_weights * fused_tokens,
            dim=1
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        out = self.classifier(fused)

        return out, sigma_r, sigma_m
