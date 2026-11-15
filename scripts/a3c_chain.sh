#!/bin/bash
cd /home/minum/26CS/MoNaVLA
for v in F30 S30 F50; do HF_HUB_OFFLINE=1 VARIANT=$v SEED=0 .venv/bin/python3 scripts/a3c_train.py >> docs/v5/egvla2_fm/a3c/train.log 2>&1; done
echo CHAIN_DONE >> docs/v5/egvla2_fm/a3c/train.log
