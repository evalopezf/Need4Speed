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

ALGORITHM="PPO"
VEHICLES=20

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME=RUN_NAME="PPO-hyperparams-${TASK_ID}"

mkdir -p "$ROOT_DIR/agentes"

python train_parsed.py \
    --EXPERIMENT_NAME PPO-100_episodes-fast_env \
    --TRAIN_EPISODES 100 \
    --N_COMBINACIONES 10 \
    --N_ENVS 20 \
    --n_steps 20 \
    --batch_size 50 \
    --gae_lambda 0.95 \
    --learning_rate_range 1e-5 1e-3 \
    --gamma_range 0.90 0.999
