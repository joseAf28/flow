import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatures(nn.Module):
    def __init__(self, dim, scale=10.0):
        super().__init__()
        self.register_buffer('B', scale * torch.randn(dim//2, 1))
    def forward(self, t):
        # t: [B] → [B×1] @ [1×(dim/2)] → [B×(dim/2)]
        t = t.unsqueeze(1)
        proj = 2*torch.pi * t @ self.B.T
        return torch.cat([proj.sin(), proj.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm  = nn.GroupNorm(8, ch)
    def forward(self, x):
        h = self.norm(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class ResNetFlow(nn.Module):
    def __init__(self, depth=18, base_ch=64, time_dim=128, num_classes=None):
        super().__init__()
        # time embed
        self.time_mlp = nn.Sequential(
            FourierFeatures(time_dim, scale=10.0),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # initial conv
        self.conv0 = nn.Conv2d(3, base_ch, 3, padding=1)
        # build ResNet layers
        layers = []
        channels = [base_ch, base_ch*2, base_ch*4, base_ch*8]
        blocks_per_stage = {18:[2,2,2,2],34:[3,4,6,3]}[depth]
        in_ch = base_ch
        for out_ch, n_blocks in zip(channels, blocks_per_stage):
            stage = []
            for i in range(n_blocks):
                stage.append(ResBlock(in_ch))
            layers.append(nn.Sequential(*stage))
            if out_ch != in_ch:
                layers.append(nn.Conv2d(in_ch, out_ch, 1))
            in_ch = out_ch
        self.resnet = nn.Sequential(*layers)
        # final conv back to 3 channels
        self.final = nn.Conv2d(in_ch, 3, 3, padding=1)


    def forward(self, x, t):
        """
        x: [B,3,32,32], t: [B]
        """
        # time broadcast
        te = self.time_mlp(t)  # [B, time_dim]
        # expand to feature maps and add
        B, _, H, W = x.shape
        te_map = te[..., None, None].expand(-1, -1, H, W)
        h = self.conv0(x) + te_map[:, :self.conv0.out_channels]
        h = self.resnet(h + te_map[:, :h.shape[1]])
        return self.final(h + te_map[:, :h.shape[1]])
