import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_train import preprocess
from tabsyn.vae.model import Encoder_model

D_TOKEN = 4
N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2

def main(args):
    dataname = args.dataname
    data_dir = f'data/{dataname}'
    device = args.device

    info_path = f'{data_dir}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/tabsyn/vae/ckpt/{dataname}'
    
    # 1. Load Data
    print(f"Loading data for {dataname}...")
    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type=info['task_type'])
    
    X_train_num, _ = X_num
    X_train_cat, _ = X_cat
    
    # 2. Marginalize Data (Independently shuffle each column to break correlations)
    print("Marginalizing data (shuffling columns independently)...")
    N = X_train_num.shape[0] if X_train_num is not None else X_train_cat.shape[0]
    
    if X_train_num is not None:
        marginal_num = np.zeros_like(X_train_num)
        for i in range(X_train_num.shape[1]):
            perm = np.random.permutation(N)
            marginal_num[:, i] = X_train_num[perm, i]
        marginal_num = torch.tensor(marginal_num).float().to(device)
    else:
        marginal_num = None

    if X_train_cat is not None:
        marginal_cat = np.zeros_like(X_train_cat)
        for i in range(X_train_cat.shape[1]):
            perm = np.random.permutation(N)
            marginal_cat[:, i] = X_train_cat[perm, i]
        marginal_cat = torch.tensor(marginal_cat).to(device)
    else:
        marginal_cat = None

    # 3. Load pre-trained VAE Model
    print("Loading pre-trained VAE encoder...")
    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_encoder.load_state_dict(torch.load(f'{ckpt_dir}/encoder.pt', map_location=device))
    pre_encoder.eval()

    # 4. Encode Marginalized Data
    print("Encoding marginalized data to latent space...")
    with torch.no_grad():
        train_z_marginal = pre_encoder(marginal_num, marginal_cat).detach().cpu().numpy()

    # 5. Save
    save_path = f'{ckpt_dir}/train_z_marginal.npy'
    np.save(save_path, train_z_marginal)
    print(f"Successfully saved marginalized latents to {save_path}!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Marginal Data')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    main(args)
