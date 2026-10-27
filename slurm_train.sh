#!/usr/bin/env bash
#SBATCH --job-name=need4speed
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%A_%a.out

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Si el cluster utiliza modules, carga aqui Python y el entorno virtual.
# module load python
# source .venv/bin/activate

RUN_NAMES=(
    "DQN-fast-20-cars"
    "PPO-fast-20-cars"
    "A2C-fast-20-cars"
    "DQN-fast-no-cars"
)
ALGORITHMS=("DQN" "PPO" "A2C" "DQN")
VEHICLES_COUNT=(20 20 20 0)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="${RUN_NAMES[$TASK_ID]}"
ALGORITHM="${ALGORITHMS[$TASK_ID]}"
VEHICLES="${VEHICLES_COUNT[$TASK_ID]}"

mkdir -p logs agentes

python3 slurm_train.py \
    --algorithm "$ALGORITHM" \
    --env-name highway-fast-v0 \
    --vehicles-count "$VEHICLES" \
    --experiment-name "$RUN_NAME" \
    --train-episodes 100 \
    --n-combinations 10 \
    --eval-episodes 10 \
    --seed 42
