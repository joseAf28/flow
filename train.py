import os
import hydra
from omegaconf import DictConfig
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import hydra.utils as hu
from data.toy import GaussianMixture2D
from models.vector_field import ToyVectorField

def rk4_step(f, x0, t0, dt):
    k1 = f(x0, t0)
    k2 = f(x0 + 0.5 * dt * k1, t0 + 0.5 * dt)
    k3 = f(x0 + 0.5 * dt * k2, t0 + 0.5 * dt)
    k4 = f(x0 + dt * k3, t0 + dt)
    return x0 + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # Determine device
    device = torch.device(cfg.device if torch.backends.mps.is_available() else 'cpu')

    # Setup output directory
    save_dir = hu.get_original_cwd() + '/outputs'
    os.makedirs(save_dir, exist_ok=True)

    # Dataset & Dataloader
    ds = GaussianMixture2D(
        n_samples=cfg.dataset.n_samples,
        n_modes=cfg.dataset.n_modes,
        radius=cfg.dataset.radius,
        std=cfg.dataset.std,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )

    # Model & Optimizer
    model = ToyVectorField(
        hidden_dim=cfg.model.hidden_dim,
        time_embed_dim=cfg.model.time_embed_dim,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.training.epochs)
    
    
    best_loss = float('inf')
    ckpt_path = os.path.join(save_dir, cfg.training.best_model_name)
    
    # Training Loop
    for epoch in range(1, cfg.training.epochs + 1):
        epoch_loss = 0.0
        for x0 in loader:
            x0 = x0.to(device)
            x1 = torch.randn_like(x0, device=device)
            t  = torch.rand(x0.size(0), device=device)
            xt = (1 - t)[:, None] * x0 + t[:, None] * x1

            v_pred = model(xt, t)
            loss   = nn.functional.mse_loss(v_pred, x1 - x0)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * x0.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(ds)
        if epoch % cfg.training.log_every == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")
        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)

if __name__ == '__main__':
    main()
