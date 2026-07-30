#!/bin/bash
# TabDrift End-to-End Execution Pipeline (Native 1-Step Drifting Models)

DATANAME=${1:-adult}
EPOCHS=${2:-500}

VAE_EMB="tabsyn/vae/ckpt/$DATANAME/train_z.npy"

if [ -f "$VAE_EMB" ]; then
    echo "=== Step 1: VAE embeddings found at $VAE_EMB. Skipping VAE re-training. ==="
else
    echo "=== Step 1: Training VAE ==="
    python main.py --dataname $DATANAME --method vae --mode train
fi

echo "=== Step 2: Training TabDrift Model (1-Step Drifting Generator, $EPOCHS epochs) ==="
PYTHONPATH=. python tabsyn/drift_train.py \
    --dataname $DATANAME \
    --gpu -1 \
    --epochs $EPOCHS \
    --batch_size 4096 \
    --lr 1e-4 \
    --hidden_size 1024 \
    --temperatures 0.1 0.5 1.0 2.0 \
    --patience 150

echo "=== Step 3: Generating Synthetic Data in 1-Step (1-NFE) ==="
PYTHONPATH=. python tabsyn/drift_sample.py \
    --dataname $DATANAME \
    --gpu -1

echo "Pipeline finished! Synthetic dataset saved in synthetic/$DATANAME/tabdrift.csv"
