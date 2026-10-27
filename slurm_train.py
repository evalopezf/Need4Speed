#!/usr/bin/env python3
"""Entrenamiento de highway-env con Optuna, preparado para Slurm.

Ejemplo:
    python slurm_train.py --algorithm DQN --vehicles-count 20 \
        --experiment-name DQN-20-cars --n-combinations 10
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import optuna
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

ALGORITHMS = {"A2C": A2C, "DQN": DQN, "PPO": PPO}

# Modifica estos rangos para definir el estudio de Optuna.
OPTUNA_CONFIG = {
    "learning_rate": (1e-5, 1e-3),
    "gamma": (0.90, 0.999),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="DQN")
    parser.add_argument("--env-name", default="highway-fast-v0")
    parser.add_argument("--vehicles-count", type=int, default=20)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--n-combinations", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("agentes"))
    return parser.parse_args()


def make_env(env_name: str, vehicles_count: int, seed: int | None = None):
    env = gym.make(
        env_name,
        render_mode="rgb_array",
        config={"vehicles_count": vehicles_count},
    )
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def suggest_hyperparameters(trial: optuna.Trial, algorithm: str) -> dict:
    learning_rate = trial.suggest_float(
        "learning_rate",
        OPTUNA_CONFIG["learning_rate"][0],
        OPTUNA_CONFIG["learning_rate"][1],
        log=True,
    )
    gamma = trial.suggest_float(
        "gamma",
        OPTUNA_CONFIG["gamma"][0],
        OPTUNA_CONFIG["gamma"][1],
    )

    params = {
        "learning_rate": learning_rate,
        "gamma": gamma,
    }

    if algorithm == "DQN":
        params.update(
            buffer_size=trial.suggest_categorical("buffer_size", [5_000, 10_000, 50_000]),
            learning_starts=trial.suggest_categorical("learning_starts", [100, 500]),
            batch_size=trial.suggest_categorical("batch_size", [32, 64]),
            target_update_interval=trial.suggest_categorical(
                "target_update_interval", [250, 500, 1_000]
            ),
        )
    else:
        params.update(
            n_steps=trial.suggest_categorical("n_steps", [64, 128, 256]),
            ent_coef=trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True),
        )
        if algorithm == "PPO":
            params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64])

    return params


def evaluate_model(model, env_name: str, vehicles_count: int, episodes: int, seed: int):
    rewards = []
    eval_env = make_env(env_name, vehicles_count)
    try:
        for episode in range(episodes):
            eval_env.reset(seed=seed + episode)
            episode_reward, _ = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=1,
                deterministic=True,
                return_episode_rewards=True,
            )
            rewards.append(float(episode_reward[0]))
    finally:
        eval_env.close()
    return float(np.mean(rewards))


def main() -> None:
    args = parse_args()
    if args.vehicles_count < 0:
        raise ValueError("vehicles-count no puede ser negativo")

    random.seed(args.seed)
    np.random.seed(args.seed)

    experiment_dir = args.output_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    timesteps_per_episode = 30 if args.env_name == "highway-fast-v0" else 40
    total_timesteps = args.train_episodes * timesteps_per_episode
    algorithm_class = ALGORITHMS[args.algorithm]
    best_info = {}
    start_time = time.time()

    def objective(trial: optuna.Trial) -> float:
        hyperparameters = suggest_hyperparameters(trial, args.algorithm)
        train_env = make_env(args.env_name, args.vehicles_count, args.seed)
        try:
            model = algorithm_class(
                "MlpPolicy",
                train_env,
                seed=args.seed,
                verbose=0,
                **hyperparameters,
            )
            model.learn(total_timesteps=total_timesteps)
            mean_reward = evaluate_model(
                model,
                args.env_name,
                args.vehicles_count,
                args.eval_episodes,
                args.seed + 1000,
            )

            completed_trials = trial.study.get_trials(
                states=(optuna.trial.TrialState.COMPLETE,)
            )
            previous_best = max(
                (completed.value for completed in completed_trials if completed.value is not None),
                default=None,
            )
            if previous_best is None or mean_reward > previous_best:
                model.save(experiment_dir / "best_model")
                best_info.update(
                    mean_reward=mean_reward,
                    trial_number=trial.number,
                    hyperparameters=hyperparameters,
                )

            return mean_reward
        finally:
            train_env.close()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_combinations, n_jobs=1)

    duration = time.time() - start_time
    summary = {
        "experiment": args.experiment_name,
        "algorithm": args.algorithm,
        "environment": args.env_name,
        "vehicles_count": args.vehicles_count,
        "train_episodes": args.train_episodes,
        "combinations": args.n_combinations,
        "duration_seconds": round(duration, 2),
        "best_reward": study.best_value,
        "best_params": study.best_params,
        "best_trial": best_info.get("trial_number", study.best_trial.number),
    }
    with (experiment_dir / "info_train.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Modelo guardado en: {experiment_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()
