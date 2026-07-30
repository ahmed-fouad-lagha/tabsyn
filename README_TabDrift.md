# TabDrift: Generative Tabular Modeling via Drifting

TabDrift adapts the recently introduced **Drifting Models** paradigm to the domain of mixed-type tabular data generation. 

By mapping noise directly to a continuous VAE latent space in a single forward pass (1-NFE), TabDrift achieves generation speeds >50x faster than traditional score-based diffusion models (e.g., TabSyn, TabDDPM). 

More importantly, the **Attraction-Repulsion Mean-Shift Kernel** naturally enforces discrete mode separation in the latent space, preventing the "categorical mode collapse" that plagues other 1-step tabular generators.

## Usage

**1. Train the VAE**
```bash
python main.py --dataname adult --method vae --mode train
```

**2. Train the TabDrift Model**
```bash
PYTHONPATH=. python tabsyn/drift_train.py --dataname adult --epochs 4000
```

**3. Generate Synthetic Data in 1-Step**
```bash
PYTHONPATH=. python tabsyn/drift_sample.py --dataname adult
```
