#!/usr/bin/env bash
#$ -N weather_riesz
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 1
#$ -l tmem=10G
#$ -l h_rt=06:00:00
#$ -R y
#$ -j y
set -euo pipefail
# Submit from the experiment directory after activating a compatible environment.
# The scheduler supplies CUDA_VISIBLE_DEVICES; do not override it.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export DRIFT_COMPILE=0
test -f "${WEATHER_DATA:-data/temperature10_group0.npz}"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
bash scripts/run_comparison.sh
