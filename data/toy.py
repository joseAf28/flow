from torch.utils.data import Dataset
import torch

def make_mixture_centers(n_modes:int, radius:float) -> torch.Tensor:
    thetas = torch.linspace(0, 2 * torch.pi, steps=n_modes +1)[:-1]
    return torch.stack([radius * torch.cos(thetas), radius * torch.sin(thetas)], dim=1)



class GaussianMixture2D(Dataset):
    
    def __init__(
        self, 
        n_samples:int = 100_000,
        n_modes: int = 8,
        radius: float = 4.0,
        std: float = 0.2
    ):
        super().__init__()
        self.data = self._generate(n_samples, n_modes, radius, std)
    
    
    def _generate(self, n_samples, n_modes, radius, std) -> torch.Tensor:
        centers = make_mixture_centers(n_modes, radius)
        idx = torch.randint(0, n_modes, (n_samples,))
        samples = centers[idx] + std * torch.randn(n_samples, 2)
        return samples
    
    
    def __len__(self) -> int:
        return len(self.data)
    
    
    def __getitem__(self, idx:int) -> torch.Tensor:
        return self.data[idx]