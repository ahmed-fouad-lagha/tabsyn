import os
import math
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import warnings
import time
from tqdm import tqdm

from tabsyn.latent_utils import get_input_train

warnings.filterwarnings('ignore')


def compute_drifting_field(x, y_pos, y_neg, temperatures, drift_scale=1.5):
    """
    Computes the Drifting Field V(x) based on Algorithm 2 and Eq. (11) of Deng et al. (2025).
    
    Args:
        x: Generated samples [N, D]
        y_pos: Real data samples [N_pos, D]
        y_neg: Generated samples [N_neg, D] (detached)
        temperatures: List of temperature scalars for the multi-scale kernel.
        drift_scale: Drift multiplier factor c (default: 1.5)
    """
    N, D = x.shape
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]

    scale_factor = math.sqrt(D)

    def pairwise_dist(a, b):
        a_norm = (a ** 2).sum(1).view(-1, 1)
        b_norm = (b ** 2).sum(1).view(1, -1)
        dist_sq = a_norm + b_norm - 2.0 * torch.mm(a, b.t())
        return dist_sq.clamp(min=1e-12).sqrt()

    dist_pos = pairwise_dist(x, y_pos)  # [N, N_pos]
    dist_neg = pairwise_dist(x, y_neg)  # [N, N_neg]

    eye = torch.eye(N, device=x.device)
    dist_neg = dist_neg + eye * 1e6

    total_drift = torch.zeros_like(x)

    for T in temperatures:
        T_scaled = T * scale_factor

        logit_pos = -dist_pos / T_scaled
        logit_neg = -dist_neg / T_scaled

        logits = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]

        # Doubly Stochastic Kernel Normalization (Algorithm 2)
        A_row = torch.softmax(logits, dim=1)
        A_col = torch.softmax(logits, dim=0)
        A = torch.sqrt(A_row * A_col)
        del logits, A_row, A_col

        A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=1)
        del A

        term_y_pos = torch.mm(A_pos, y_pos)
        term_y_neg = torch.mm(A_neg, y_neg)

        sum_w_pos = A_pos.sum(dim=1, keepdim=True)
        sum_w_neg = A_neg.sum(dim=1, keepdim=True)
        del A_pos, A_neg

        term_x_pos = sum_w_pos * x
        term_x_neg = sum_w_neg * x

        V_pos = term_y_pos - term_x_pos
        V_neg = term_y_neg - term_x_neg

        total_drift += (V_pos - V_neg)
        del V_pos, V_neg, term_y_pos, term_y_neg, term_x_pos, term_x_neg

    return total_drift * drift_scale


class ResBlock(nn.Module):
    """
    Residual Block with LayerNorm, SiLU, and Dropout for TabDrift Generator.
    Matches TabSyn Score Network parameter capacity (~10.2M parameters).
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)


class TabDriftGenerator(nn.Module):
    """
    Unconditioned Residual Generator for TabDrift.
    Maps Gaussian noise to continuous VAE latent space with residual depth (~10.2M params).
    """
    def __init__(self, z_dim, hidden_size=1024, num_res_blocks=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(z_dim, hidden_size)
        self.blocks = nn.ModuleList([
            ResBlock(hidden_size, dropout=dropout) for _ in range(num_res_blocks)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, z_dim)
        )

    def forward(self, noise):
        x = self.in_proj(noise)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)


def main(args):
    device = args.device
    train_z, _, _, ckpt_path, _ = get_input_train(args)

    drift_ckpt_path = f'{ckpt_path}/drift_model.pt'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1]
    mean = train_z.mean(0)
    # Match TabSyn's normalization: (z - mean) / 2
    # This preserves the VAE's learned scale hierarchy across dimensions
    train_data = (train_z - mean) / 2

    # Save normalization stats and model config
    torch.save({'mean': mean}, f'{ckpt_path}/drift_norm.pt')
    config = {
        'in_dim': in_dim,
        'hidden_size': args.hidden_size,
        'num_res_blocks': 4,
        'temperatures': args.temperatures,
        'dataname': args.dataname
    }
    with open(f'{ckpt_path}/drift_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    batch_size = args.batch_size
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0 if device == 'cpu' else 4, drop_last=True
    )

    model = TabDriftGenerator(z_dim=in_dim, hidden_size=args.hidden_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # EMA model for stable inference (decay=0.999)
    ema_model = copy.deepcopy(model)
    ema_decay = 0.999

    num_epochs = args.epochs
    temperatures = args.temperatures

    best_loss = float('inf')
    patience_counter = 0

    print(f'Training TabDrift Generator for {num_epochs} epochs on {device}...')
    print(f'Kernel temperatures: {temperatures}, Hidden size: {args.hidden_size}, Batch size: {batch_size}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, total=len(train_loader), leave=False)
        pbar.set_description(f'Epoch {epoch+1}/{num_epochs}')

        for z_batch in pbar:
            z_batch = z_batch.float().to(device)
            optimizer.zero_grad()

            noise = torch.randn_like(z_batch)
            fake_z = model(noise)

            y_pos = z_batch
            y_neg = fake_z.detach()

            V = compute_drifting_field(fake_z, y_pos, y_neg, temperatures, drift_scale=args.drift_scale)

            target = (fake_z + V).detach()

            loss = F.mse_loss(fake_z, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            epoch_loss += loss.item() * len(z_batch)
            pbar.set_postfix({'Loss': f'{loss.item():.4e}'})

        scheduler.step()
        curr_loss = epoch_loss / len(train_data)

        # Update EMA model
        with torch.no_grad():
            for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1 - ema_decay)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience_counter = 0
            torch.save(model.state_dict(), drift_ckpt_path)
            torch.save(ema_model.state_dict(), f'{ckpt_path}/drift_model_ema.pt')
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            curr_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch+1}/{num_epochs}: Loss={curr_loss:.6f}, Best={best_loss:.6f}, LR={curr_lr:.6e}'
            )

        if args.patience > 0 and patience_counter >= args.patience:
            print(f'Early stopping triggered after {epoch+1} epochs (no improvement for {args.patience} epochs).')
            break

    print(f'Training Complete. Total Time: {time.time() - start_time:.2f}s')
    print(f'Saved Model Checkpoint to {drift_ckpt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TabDrift Generative Model Training')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index (-1 for CPU).')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size (1024 matches TabSyn ~10M params).')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping norm.')
    parser.add_argument('--drift_scale', type=float, default=1.5, help='Drift scale factor multiplier c (default: 1.5).')
    parser.add_argument('--patience', type=int, default=250, help='Early stopping patience epochs.')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.1, 0.5, 1.0, 2.0], help='Multi-scale kernel temperatures (calibrated for tabular VAE latents).')
    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)

