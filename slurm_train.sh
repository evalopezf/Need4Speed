#!/bin/bash
#
#SBATCH -p special
#SBATCH --array=0-7%8
#SBATCH -J hackaton_prueba
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH -o /lustre/home/cebas/emlopez/hackaton/slurm-%A_%a.out
#SBATCH --error=/lustre/home/cebas/emlopez/hackaton/slurm-%A_%a.err
set -euo pipefail

ROOT_DIR="/lustre/home/cebas/emlopez/hackaton/"
cd "$ROOT_DIR"

module load rama0.4 GCCcore/14.2.0 Python/3.13.1

VENV_DIR="/lustre/home/cebas/emlopez/hackaton/.venv-slurm"
VENV_LOCK="/lustre/home/cebas/emlopez/hackaton/.venv-slurm.lock"
exec 9>"$VENV_LOCK"
flock -x 9

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --disable-pip-version-check -r "$ROOT_DIR/requirements-slurm.txt"
flock -u 9

RUN_NAME="PPO-hyperparams"
ALGORITHM="PPO"
VEHICLES=20

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$ROOT_DIR/agentes"

python slurm_train.py \
    --algorithm "$ALGORITHM" \
    --env-name highway-fast-v0 \
    --vehicles-count "$VEHICLES" \
    --experiment-name "$RUN_NAME" \
    --train-episodes 100 \
    --trials-per-task 1 \
    --eval-episodes 10 \
    --study-name "ppo_highway" \
    --storage-path "$ROOT_DIR/agentes/optuna_ppo.db" \
    --seed $((42 + TASK_ID)) \
    --output-dir "$ROOT_DIR/agentes"
