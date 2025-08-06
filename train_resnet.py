import os, math, argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
import wandb

from data.cifar10_resnet import get_cifar10_loaders
from models.resnet_flow import ResNetFlow

def rk4_step(f, x0, t0, dt):
    k1 = f(x0, t0)
    k2 = f(x0 + 0.5*dt.unsqueeze(1)*k1, t0 + 0.5*dt)
    k3 = f(x0 + 0.5*dt.unsqueeze(1)*k2, t0 + 0.5*dt)
    k4 = f(x0 + dt.unsqueeze(1)*k3, t0 + dt)
    return x0 + (dt.unsqueeze(1)/6)*(k1 + 2*k2 + 2*k3 + k4)

def train(args):
    # reproducibility
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_loader, _ = get_cifar10_loaders(args.batch_size, args.workers, args.data_root)

    # model & optimizer
    model = ResNetFlow(depth=args.depth, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()
    best_loss = math.inf

    # wandb init
    wandb.init(project="flow-matching-resnet", config=vars(args))

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x0 in pbar:
            x0 = x0.to(device)
            # sample prior + t
            x1 = torch.randn_like(x0)
            t  = torch.rand(x0.size(0), device=device)

            xt = (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*x1

            opt.zero_grad()
            with autocast():
                v_pred = model(xt, t)
                loss = F.mse_loss(v_pred, x1 - x0)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * x0.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader.dataset)
        wandb.log({"epoch": epoch, "train_loss": avg_loss})

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.save_dir, "best_resnet.pt")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            wandb.log({"best_loss": best_loss})
            print(f"→ New best ({best_loss:.4f}), saved to {ckpt_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str,   default="data/")
    p.add_argument("--save_dir",     type=str,   default="/content/drive/MyDrive/flow_runs")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--depth",        type=int,   choices=[18,34], default=18)
    p.add_argument("--base_ch",      type=int,   default=64)
    p.add_argument("--time_dim",     type=int,   default=128)
    p.add_argument("--mixed_precision", action="store_true")
    args = p.parse_args()
    train(args)
