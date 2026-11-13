import torch
import torch.nn as nn
from torchvision import models
from modified_cbam import ModifiedCBAM

# ============================================================
# BLOCK 5: MobileNetV2 + CBAM BRANCH
# ------------------------------------------------------------
# This branch:
# - Uses pretrained MobileNetV2
# - Extracts feature maps (NOT classifier output)
# - Freezes most layers
# - Fine-tunes last stage
# - Applies CBAM attention
# ============================================================

class MobileNetCBAM(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Load pretrained MobileNetV2
        # ----------------------------------------------------
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # Extract feature layers
        self.features = base.features

        # ----------------------------------------------------
        # Freeze all layers
        # ----------------------------------------------------
        for param in self.features.parameters():
            param.requires_grad = False

        # ----------------------------------------------------
        # Unfreeze last stage (layers 14 onward)
        # ----------------------------------------------------
        for param in self.features[14:].parameters():
            param.requires_grad = True

        # ----------------------------------------------------
        # CBAM Attention
        # MobileNetV2 output channels = 1280
        # ----------------------------------------------------
        self.cbam = ModifiedCBAM(1280)

    def forward(self, x):
        """
        Input:
            x → (B, 3, H, W)

        Output:
            feature map → (B, 1280, H', W')
        """

        # Forward through feature layers
        for layer in self.features:
            x = layer(x)

        # Apply CBAM attention
        x = self.cbam(x)

        return x
