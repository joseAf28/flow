import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, mapping_size: int = 64, scale: float = 10.0) -> None:
        super().__init__()
        self.register_buffer('B', scale * torch.randn(mapping_size, 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Broadcast scalar t to a 1D tensor if needed
        if t.dim() == 0:
            t = t.unsqueeze(0)
        proj = 2 * torch.pi * t.unsqueeze(1) @ self.B.T
        return torch.cat([proj.sin(), proj.cos()], dim=1)

class ToyVectorField(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_embed_dim: int = 128
    ) -> None:
        super().__init__()
        self.time_embed = FourierFeatures(mapping_size=time_embed_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(2 + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Ensure time tensor is 1D matching batch size
        if t.dim() == 0:
            t = t.expand(x.size(0))
        te = self.time_embed(t)
        return self.net(torch.cat([x, te], dim=1))