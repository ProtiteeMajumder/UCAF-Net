import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# BLOCK 6: CHANNEL ALIGNMENT LAYERS
# ============================================================

class FeatureAlignment(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Align ResNet features: 2048 → 512
        # ----------------------------------------------------
        self.align_r = nn.Conv2d(
            in_channels=2048,
            out_channels=512,
            kernel_size=1
        )

        # ----------------------------------------------------
        # Align MobileNet features: 1280 → 512
        # ----------------------------------------------------
        self.align_m = nn.Conv2d(
            in_channels=1280,
            out_channels=512,
            kernel_size=1
        )

    def forward(self, fr, fm):

        fr = self.align_r(fr)
        fm = self.align_m(fm)

        return fr, fm

# ============================================================
# BLOCK 7: UNCERTAINTY-AWARE CROSS ATTENTION
# ------------------------------------------------------------
# PURPOSE:
# - Enable interaction between two models
# - Suppress unreliable (high-σ) tokens
# ============================================================

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # Linear projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        # scaling for stability
        self.scale = dim ** -0.5

    def forward(self, x_q, x_kv, sigma):
        """
        x_q  : (B, N, C) → Query
        x_kv : (B, N, C) → Key & Value
        sigma: (B, N)    → uncertainty
        """

        # Step 1: Q, K, V
        Q = self.q(x_q)
        K = self.k(x_kv)
        V = self.v(x_kv)

        # Step 2: similarity matrix
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Step 3: uncertainty penalty
        sigma = sigma.unsqueeze(1)   # (B,1,N)
        attn = attn - sigma

        # Step 4: normalize
        attn = torch.softmax(attn, dim=-1)

        # Step 5: weighted aggregation
        out = torch.matmul(attn, V)

        return out

# ============================================================
# BLOCK 8: UNCERTAINTY HEAD
# ------------------------------------------------------------
# PURPOSE:
# - Predict per-token uncertainty σ
# - Learn which regions are unreliable
# ============================================================

class UncertaintyHead(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # map feature → scalar uncertainty
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x: (B, N, C)

        returns:
        sigma: (B, N)
        """

        sigma = self.fc(x).squeeze(-1)   # (B,N)

        # ensure positivity
        sigma = torch.exp(sigma)

        return sigma

# ============================================================
# BLOCK 9: UNCERTAINTY-GUIDED GATED FUSION
# ------------------------------------------------------------
# PURPOSE:
# - Adaptive fusion between ResNet and MobileNet
# - Uses uncertainty to guide fusion confidence
# ============================================================

class GatedFusion(nn.Module):

    def __init__(self, dim):
        super().__init__()

        # ----------------------------------------------------
        # Gate generator
        # Input:
        #   feature concat + uncertainty concat
        # ----------------------------------------------------

        self.fc = nn.Sequential(

            nn.Linear(dim * 2 + 2, dim),

            nn.ReLU(inplace=True),

            nn.Linear(dim, dim)
        )

    def forward(self, fr, fm, sigma_r, sigma_m):
        """
        fr       : ResNet features      (B,N,C)
        fm       : MobileNet features   (B,N,C)

        sigma_r  : ResNet uncertainty   (B,N)
        sigma_m  : MobileNet uncertainty(B,N)
        """

        # ====================================================
        # FEATURE CONCATENATION
        # ====================================================

        feature_concat = torch.cat([fr, fm], dim=-1)

        # ====================================================
        # UNCERTAINTY CONCATENATION
        # ====================================================

        sigma_r = sigma_r.unsqueeze(-1)   # (B,N,1)
        sigma_m = sigma_m.unsqueeze(-1)

        uncertainty_concat = torch.cat(
            [sigma_r, sigma_m],
            dim=-1
        )  # (B,N,2)

        # ====================================================
        # COMBINE FEATURES + UNCERTAINTY
        # ====================================================

        combined = torch.cat(
            [feature_concat, uncertainty_concat],
            dim=-1
        )

        # ====================================================
        # GATE COMPUTATION
        # ====================================================

        gate = torch.sigmoid(self.fc(combined))

        # ====================================================
        # UNCERTAINTY-GUIDED FUSION
        # ====================================================

        fused = gate * fr + (1 - gate) * fm

        return fused
