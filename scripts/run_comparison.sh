#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_device=${WEATHER_DEVICE:-cuda}
task_data=${WEATHER_DATA:-data/temperature10_group0.npz}
task_runs=${WEATHER_RUNS:-runs/temperature10_group0}
for task_method in energy riesz_softmax; do
  task_checkpoints=()
  for task_seed in 42 43 44 45 46 47 48 49 50 51; do
    task_work="$task_runs/$task_method/seed$task_seed"
    task_args=(--data "$task_data" --method "$task_method" --seed "$task_seed"
               --device "$task_device" --workdir "$task_work")
    if [[ -f "$task_work/latest.pt" ]]; then task_args+=(--resume); fi
    python -u train.py "${task_args[@]}"
    task_checkpoints+=("$task_work/best.pt")
  done
  if [[ ! -f "$task_runs/$task_method/evaluation/metrics.json" ]]; then
    python -u evaluate.py --data "$task_data" --device "$task_device" \
      --checkpoints "${task_checkpoints[@]}" --output "$task_runs/$task_method/evaluation"
  fi
done
