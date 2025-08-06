# models/resnet_flow.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatures(nn.Module):
    """Maps a scalar t ∈ [0,1] to a high-dim Fourier feature vector."""
    def __init__(self, dim: int, scale: float = 10.0):
        super().__init__()
        # We want an output of size `dim`. Half sin, half cos ⇒ mapping_size = dim//2
        self.register_buffer('B', scale * torch.randn(dim // 2, 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        t = t.unsqueeze(1)                           # → [B, 1]
        proj = 2 * torch.pi * t @ self.B.T          # → [B, dim//2]
        return torch.cat([proj.sin(), proj.cos()], dim=1)  # → [B, dim]


class ResBlock(nn.Module):
    """A simple residual block: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + skip."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class ResNetFlow(nn.Module):
    """
    ResNet-style vector field for flow-matching.
    - depth: 18 or 34 (defines blocks per stage)
    - base_ch: number of channels in first conv; doubles each stage
    - time_dim: dimension of your time MLP (we project to each stage's channel count)
    """
    def __init__(self, depth: int = 18, base_ch: int = 64, time_dim: int = 128):
        super().__init__()
        # 1) time MLP producing a `time_dim` vector
        self.time_mlp = nn.Sequential(
            FourierFeatures(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # 2) per-stage projections from `time_dim` → channel count
        self.time_proj0 = nn.Linear(time_dim, base_ch)
        self.time_proj1 = nn.Linear(time_dim, base_ch)
        self.time_proj2 = nn.Linear(time_dim, base_ch * 2)
        self.time_proj3 = nn.Linear(time_dim, base_ch * 4)
        self.time_proj4 = nn.Linear(time_dim, base_ch * 8)

        # 3) initial conv
        self.conv0 = nn.Conv2d(3, base_ch, kernel_size=3, padding=1)

        # 4) build ResNet stages
        blocks_cfg = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}[depth]
        b1, b2, b3, b4 = blocks_cfg

        # Stage 1: stays at base_ch
        self.layer1 = nn.Sequential(*[ResBlock(base_ch) for _ in range(b1)])

        # Stage 2: increase to 2× channels
        self.layer2_down = nn.Conv2d(base_ch, base_ch * 2, kernel_size=1)
        self.layer2      = nn.Sequential(*[ResBlock(base_ch * 2) for _ in range(b2)])

        # Stage 3: increase to 4× channels
        self.layer3_down = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=1)
        self.layer3      = nn.Sequential(*[ResBlock(base_ch * 4) for _ in range(b3)])

        # Stage 4: increase to 8× channels
        self.layer4_down = nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=1)
        self.layer4      = nn.Sequential(*[ResBlock(base_ch * 8) for _ in range(b4)])

        # 5) final conv back to 3 channels
        self.final = nn.Conv2d(base_ch * 8, 3, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W],  t: [B]
        returns: velocity field [B,3,H,W]
        """
        # Embed time once
        te = self.time_mlp(t)  # → [B, time_dim]

        # Stage 0: initial conv + time
        te0 = self.time_proj0(te)[:, :, None, None]  # → [B, base_ch,1,1]
        h   = self.conv0(x) + te0

        # Stage 1
        te1 = self.time_proj1(te)[:, :, None, None]  # [B, base_ch,1,1]
        h   = h + te1
        h   = self.layer1(h)

        # Stage 2
        te2 = self.time_proj2(te)[:, :, None, None]  # [B, base_ch*2,1,1]
        h   = self.layer2_down(h) + te2
        h   = self.layer2(h)

        # Stage 3
        te3 = self.time_proj3(te)[:, :, None, None]  # [B, base_ch*4,1,1]
        h   = self.layer3_down(h) + te3
        h   = self.layer3(h)

        # Stage 4
        te4 = self.time_proj4(te)[:, :, None, None]  # [B, base_ch*8,1,1]
        h   = self.layer4_down(h) + te4
        h   = self.layer4(h)

        # Final conv (you can also add te4 again if you like)
        return self.final(h + te4)
