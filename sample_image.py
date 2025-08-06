
import torch
import matplotlib.pyplot as plt
from models.unet import UNetVectorField   # your image-scale vector field
from sample import rk4_step               # reuse the RK4 helper



def rk4_step(f, x0, t0, dt):
    # x0: [batch, dim], t0: [batch], dt: [batch]
    k1 = f(x0, t0)
    k2 = f(x0 + 0.5 * dt.unsqueeze(1) * k1, t0 + 0.5 * dt)
    k3 = f(x0 + 0.5 * dt.unsqueeze(1) * k2, t0 + 0.5 * dt)
    k4 = f(x0 + dt.unsqueeze(1) * k3, t0 + dt)
    return x0 + (dt.unsqueeze(1) / 6) * (k1 + 2 * k2 + 2 * k3 + k4)




@torch.no_grad()
def sample_image(model, n, steps, device):
    # start from N(0,I) noise at image shape
    x = torch.randn(n, 3, 32, 32, device=device)  # e.g. CIFAR-10 32×32
    ts = torch.linspace(1.0, 0.0, steps, device=device)
    for i in range(steps-1):
        t0 = ts[i].repeat(n)
        t1 = ts[i+1].repeat(n)
        dt = t1 - t0
        x = rk4_step(lambda xx, tt: model(xx, tt), x, t0, dt)
    return x.clamp(-1,1).cpu()  # back to [–1,1]

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # instantiate & load your trained U-Net
    model = UNetVectorField(base_ch=64, ch_mult=(1,2,2,4), time_dim=256)
    ckpt = torch.load("outputs/best_cifar10.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    # sample 16 images with 64 RK4 steps
    samples = sample_image(model, n=16, steps=64, device=device)

    # plot in a 4×4 grid
    fig, axes = plt.subplots(4,4, figsize=(6,6))
    for img, ax in zip(samples, axes.flatten()):
        ax.imshow(((img.permute(1,2,0)+1)/2).numpy())
        ax.axis('off')
    plt.tight_layout()
    plt.show()
