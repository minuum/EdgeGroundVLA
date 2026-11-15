#!/bin/bash
cd /home/minum/26CS/MoNaVLA
for sd in 1 2; do for v in D10 D00 D30; do HF_HUB_OFFLINE=1 VARIANT=$v SEED=$sd .venv/bin/python3 scripts/a3c_train.py >> docs/v5/egvla2_fm/a3c/train3.log 2>&1; done; done
echo CHAIN_DONE >> docs/v5/egvla2_fm/a3c/train3.log
