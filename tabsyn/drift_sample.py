import torch
import argparse
import warnings
import time
import os
import json

from tabsyn.drift_train import TabDriftGenerator
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target

warnings.filterwarnings('ignore')

def main(args):
    dataname = args.dataname
    device = args.device
    save_path = args.save_path
    
    if not save_path:
        if not os.path.exists(f'synthetic/{dataname}'):
            os.makedirs(f'synthetic/{dataname}')
        save_path = f'synthetic/{dataname}/tabdrift.csv'

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 

    norm_path = f'{ckpt_path}/drift_norm.pt'
    config_path = f'{ckpt_path}/drift_config.json'
    drift_ckpt_path = f'{ckpt_path}/drift_model.pt'

    if not os.path.exists(norm_path) or not os.path.exists(drift_ckpt_path):
        print(f"Drift Model files not found in {ckpt_path}. Run drift_train.py first!")
        return

    # Load config if present, else fallback to args
    hidden_size = args.hidden_size
    num_res_blocks = 4
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            hidden_size = cfg.get('hidden_size', hidden_size)
            num_res_blocks = cfg.get('num_res_blocks', num_res_blocks)

    norm_stats = torch.load(norm_path, map_location=device)
    mean = norm_stats['mean'].to(device)

    model = TabDriftGenerator(z_dim=in_dim, hidden_size=hidden_size, num_res_blocks=num_res_blocks).to(device)
    
    # Prefer EMA model weights for more stable generation
    ema_path = f'{ckpt_path}/drift_model_ema.pt'
    if os.path.exists(ema_path):
        model.load_state_dict(torch.load(ema_path, map_location=device))
        print('Loaded EMA model weights for inference.')
    else:
        model.load_state_dict(torch.load(drift_ckpt_path, map_location=device))
        print('Loaded standard model weights (no EMA found).')
    model.eval()

    num_samples = train_z.shape[0] if args.num_samples is None else args.num_samples
    steps = args.steps
    print(f"Generating {num_samples} tabular samples natively in {steps}-Step ({steps}-NFE) with TabDrift...")
    print(f"Model architecture: Residual MLP (10.2M params), hidden_size={hidden_size}, z_dim={in_dim}")
    start_time = time.time()

    with torch.no_grad():
        z = torch.randn([num_samples, in_dim], device=device)
        if steps == 1:
            fake_z = model(z)
        else:
            dt = 1.0 / steps
            for step_idx in range(steps):
                target_z = model(z)
                z = z + dt * (target_z - z)
            fake_z = z
        # Unnormalize: reverse (z - mean) / 2 => z * 2 + mean
        fake_z = fake_z * 2 + mean

    syn_data = fake_z.float().cpu().numpy()
    
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device) 
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)
    
    syn_df.to_csv(save_path, index=False)
    
    end_time = time.time()
    print(f'Time: {end_time - start_time:.4f} seconds for {num_samples} samples ({steps}-NFE).')
    print('Saved generated data to {}'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Step Data Generation with TabDrift')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index (-1 for CPU).')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size fallback.')
    parser.add_argument('--steps', type=int, default=1, help='Number of Euler refinement steps (NFE). Default: 1.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of synthetic rows to generate (default: dataset size).')
    parser.add_argument('--save_path', type=str, default=None, help='Save path for synthetic CSV.')
    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    
    main(args)
