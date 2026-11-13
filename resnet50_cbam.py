from modified_cbam import ModifiedCBAM

# ============================================================
# BLOCK 4: ResNet50 + CBAM (Improved Partial Unfreezing)
# ------------------------------------------------------------
# Strategy:
# - Load pretrained ResNet50
# - Freeze early layers
# - Unfreeze layer3 + layer4
# - Apply CBAM after layer4
# ============================================================

import torch
import torch.nn as nn
from torchvision import models

class ResNet50CBAM(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Load pretrained ResNet50
        # ----------------------------------------------------
        base = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # ----------------------------------------------------
        # Backbone layers
        # ----------------------------------------------------
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # ----------------------------------------------------
        # CBAM Attention
        # ----------------------------------------------------
        self.cbam = ModifiedCBAM(2048)

        # ----------------------------------------------------
        # Freeze ALL layers first
        # ----------------------------------------------------
        for param in self.parameters():
            param.requires_grad = False

        # ----------------------------------------------------
        # Unfreeze layer3
        # ----------------------------------------------------
        for param in self.layer3.parameters():
            param.requires_grad = True

        # ----------------------------------------------------
        # Unfreeze layer4
        # ----------------------------------------------------
        for param in self.layer4.parameters():
            param.requires_grad = True

        # ----------------------------------------------------
        # Unfreeze CBAM
        # ----------------------------------------------------
        for param in self.cbam.parameters():
            param.requires_grad = True

        # ----------------------------------------------------
        # Freeze BatchNorm statistics in frozen layers
        # ----------------------------------------------------
        self.freeze_bn()

    def freeze_bn(self):
        """
        Freeze BatchNorm running statistics
        for frozen layers.
        """

        for m in self.modules():

            if isinstance(m, nn.BatchNorm2d):

                # If ALL parameters frozen
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()

    def forward(self, x):

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ----------------------------------------------------
        # Residual stages
        # ----------------------------------------------------
        x = self.layer1(x)
        x = self.layer2(x)

        # Trainable deeper stages
        x = self.layer3(x)
        x = self.layer4(x)

        # ----------------------------------------------------
        # CBAM Attention
        # ----------------------------------------------------
        x = self.cbam(x)

        return x
