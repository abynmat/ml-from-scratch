#!/usr/bin/env bash
# Reproduce every number and figure in this repo (about 35-45 min on an Apple M-series laptop).
# Author: Abyn Scaria
set -euo pipefail
cd "$(dirname "$0")"
python3 data/download.py
for p in 01_adam 02_batchnorm 03_resnet 04_vae 05_gan 06_ddpm; do
  echo "=== $p ==="
  python3 -u "$p/train.py" | tee "$p/results/run.log"
done
python3 make_overview.py
