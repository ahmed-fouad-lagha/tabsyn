#!/bin/bash
echo "Starting TabDrift MLP Training..."
PYTHONPATH=. python tabsyn/drift_train.py --dataname adult --epochs 10000

echo "Generating 1-step samples..."
PYTHONPATH=. python tabsyn/drift_sample.py --dataname adult
