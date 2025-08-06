import os
import argparse
import torch
import matplotlib.pyplot as plt
from models.vector_field import ToyVectorField

# RK4 integrator supporting vectorized t and dt
### be careful woth times, it must have the same dimensiosio of x0

def rk4_step(f, x0, t0, dt):
    # x0: [batch, dim], t0: [batch], dt: [batch]
    k1 = f(x0, t0)
    k2 = f(x0 + 0.5 * dt.unsqueeze(1) * k1, t0 + 0.5 * dt)
    k3 = f(x0 + 0.5 * dt.unsqueeze(1) * k2, t0 + 0.5 * dt)
    k4 = f(x0 + dt.unsqueeze(1) * k3, t0 + dt)
    return x0 + (dt.unsqueeze(1) / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def sample(model, n_samples, steps, device):
    # initialize from prior
    x = torch.randn(n_samples, 2, device=device)
    # time grid (from t=1 to t=0)
    ts = torch.linspace(1.0, 0.0, steps, device=device)
    for i in range(steps - 1):
        # create batch-sized time vectors
        # expand scalars to 1D tensors
        t0 = ts[i].repeat(n_samples)
        t1 = ts[i + 1].repeat(n_samples)
        dt = t1 - t0
        # RK4 update with vectorized t0 and dt
        x = rk4_step(lambda xx, tt: model(xx, tt), x, t0, dt)
    return x.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoint_epoch100.pt')
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--save_path', type=str, default=None,
                        help='Optional path to save generated points as .npy or .csv')
    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # load model
    model = ToyVectorField().to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    pts = sample(model, args.n_samples, args.steps, device)

    # optional save
    if args.save_path:
        ext = os.path.splitext(args.save_path)[1].lower()
        if ext == '.npy':
            import numpy as np
            np.save(args.save_path, pts)
        elif ext == '.csv':
            import numpy as np
            np.savetxt(args.save_path, pts, delimiter=',')
        else:
            print(f"Unknown extension {ext}, skipping save.")

    # plot
    plt.figure(figsize=(6,6))
    plt.scatter(pts[:,0], pts[:,1], s=2)
    plt.axis('equal')
    plt.title(f"Flow-matching samples (n={args.n_samples})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

