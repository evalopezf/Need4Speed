#!/usr/bin/env python3
"""Version con argparse del script de entrenamiento PPO en highway-env.

Todos los argumentos tienen por defecto los valores del script original.

Ejemplo:
    python train_parsed.py --EXPERIMENT_NAME PPO-100_episodes-fast_env \
        --TRAIN_EPISODES 100 --N_ENVS 20 --N_COMBINACIONES 10
"""
import argparse
import os
import random
import time


# Rangos de busqueda de Optuna. Modifica estos valores (o usa los argumentos) para definir el estudio.
OPTUNA_CONFIG = {
    "learning_rate": (1e-5, 1e-3),
    "gamma": (0.90, 0.999),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Entorno / experimento
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--TRAIN_ENV_NAME", type=str, default="highway-fast-v0")
    parser.add_argument("--EXPERIMENT_NAME", type=str, default="PPO-100_episodes-fast_env")

    # Variables que SI pueden ser modificadas
    parser.add_argument("--TRAIN_EPISODES", type=int, default=100)
    parser.add_argument("--N_COMBINACIONES", type=int, default=1)
    parser.add_argument("--N_ENVS", type=int, default=20)

    # Hiperparametros fijos
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="Debe estar en [0, 1] (en el script original era 15, que no es valido)")

    # Rangos de Optuna (por defecto los de OPTUNA_CONFIG)
    parser.add_argument("--learning_rate_range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=OPTUNA_CONFIG["learning_rate"])
    parser.add_argument("--gamma_range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=OPTUNA_CONFIG["gamma"])

    # Otros
    parser.add_argument("--start_method", type=str, default="fork",
                        choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--show_plots", action="store_true",
                        help="Mostrar las graficas (por defecto solo se guardan; util en Slurm)")
    return parser.parse_args()


args = parse_args()
OPTUNA_CONFIG["learning_rate"] = tuple(args.learning_rate_range)
OPTUNA_CONFIG["gamma"] = tuple(args.gamma_range)

import matplotlib
if not args.show_plots:
    matplotlib.use("Agg")  # sin display (cluster / Slurm)
from matplotlib import pyplot as plt

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO

# Fijar la semilla
seed = args.seed
random.seed(seed)

# Indica el entorno para el entrenamiento.
TRAIN_ENV_NAME = args.TRAIN_ENV_NAME

# Crear el entorno
env = gym.make(TRAIN_ENV_NAME, render_mode='rgb_array')

# Fijar una semilla para reproducibilidad
env.reset(seed=seed)

# Mostrar el estado inicial
plt.imshow(env.render())
if args.show_plots:
    plt.show()
plt.close()

# Espacio de observaciones
print("Espacio de observaciones:", env.observation_space)

# Espacio de acciones
print("Espacio de acciones:", env.action_space)

# Se cierra el entorno de ejemplo
env.close()


# Wrapper no frenar
class NonStopWrapper(gym.ActionWrapper):
    """
    Clase `NonStopWrapper`. impide que el modelo frene
    """

    # Aqui se define como se manipulan las acciones del agente
    def action(self, action):
        if action == 3:
            action = 1
        # elif action == 1:
        #     action = 4
        return action


# Aplica aqui todos los wrappers en el orden que consideres adecuado
def aplicar_wrappers(env):
    env = NonStopWrapper(env)
    return env


# Centraliza aqui la configuracion y los wrappers de todos los entornos
def make_env():
    new_env = gym.make(TRAIN_ENV_NAME, render_mode='rgb_array')
    new_env = aplicar_wrappers(new_env)
    return new_env


EXPERIMENT_NAME = args.EXPERIMENT_NAME  # Nombre del experimento para guardar resultados

# Variables fijas que NO deben ser modificadas
N_JOBS = 1  # Numero de trabajos en paralelo para Optuna
TIMESTEPS_PER_EPISODE = 30 if TRAIN_ENV_NAME == "highway-fast-v0" else 40  # Numero de pasos por episodio
EVAL_EPISODES = 10  # Numero de episodios que dura cada evaluacion

# Variables que SI pueden ser modificadas
TRAIN_EPISODES = args.TRAIN_EPISODES  # Numero de episodios de entrenamiento
N_COMBINACIONES = args.N_COMBINACIONES  # Numero de combinaciones a explorar durante la optimizacion de hiperparametros

from stable_baselines3.common.callbacks import BaseCallback
from tqdm.std import tqdm


# Callback para mostrar una barra de progreso de texto durante el entrenamiento
class TextProgressBarCallback(BaseCallback):
    """
    Barra de progreso de texto para `learn()`. Hace lo mismo que `progress_bar=True`,
    que en VS Code necesita widgets de Jupyter y puede fallar.
    """
    def _on_training_start(self):
        self.bar = tqdm(total=self.locals["total_timesteps"], unit=" pasos")

    def _on_step(self):
        # Cada llamada es un paso de cada entorno en paralelo; SB3 puede pasarse un poco del total
        self.bar.update(min(self.training_env.num_envs, self.bar.total - self.bar.n))
        return True

    def _on_training_end(self):
        self.bar.close()


import optuna
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes

import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

N_ENVS = args.N_ENVS  # roughly the number of physical CPU cores you can spare
torch.set_num_threads(1)  # avoid CPU oversubscription (workers + torch threads)


def make_train_env(n_envs=N_ENVS):
    # Each worker gets its own Monitor so episode stats are tracked per env
    return SubprocVecEnv(
        [lambda: Monitor(make_env()) for _ in range(n_envs)],
        start_method=args.start_method,
    )


# Semillas de validacion comunes a todos los trials, distintas de entrenamiento.
EVAL_SEEDS = [seed + 1000 + i for i in range(EVAL_EPISODES)]

# Diccionario para almacenar informacion sobre el mejor modelo entrenado
best_training_info = {}

# Carpeta de salida
os.makedirs(f"agentes/{EXPERIMENT_NAME}", exist_ok=True)


# Definir la funcion para el/los entrenamiento/s
def objective(trial):

    # Hiperparametros a optimizar
    learning_rate = trial.suggest_float(
        "learning_rate", OPTUNA_CONFIG["learning_rate"][0], OPTUNA_CONFIG["learning_rate"][1], log=True
    )
    gamma = trial.suggest_float(
        "gamma", OPTUNA_CONFIG["gamma"][0], OPTUNA_CONFIG["gamma"][1]
    )

    # Resto de hiperparametros (fijos)
    n_steps = args.n_steps
    batch_size = args.batch_size
    gae_lambda = args.gae_lambda

    train_env = None
    eval_env = None

    try:
        train_env = make_train_env()
        eval_env = DummyVecEnv([lambda: Monitor(make_env())])

        # Crear agente
        agent = PPO('MlpPolicy',
                    train_env,  # No modificar
                    verbose=0,  # No modificar
                    seed=seed,  # No modificar

                    # Hiperparametros a optimizar
                    learning_rate=learning_rate,
                    gamma=gamma,
                    # Resto de hiperparametros
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gae_lambda=gae_lambda,
                    )

        # Instanciar los callbacks para este trial
        stop_training_callback = StopTrainingOnMaxEpisodes(max_episodes=TRAIN_EPISODES * TIMESTEPS_PER_EPISODE, verbose=1)  # Limita el numero de episodios al indicado
        text_progress_bar_callback = TextProgressBarCallback()  # Muestra una barra de progreso de texto durante el entrenamiento

        # Lista de callbacks. Incluye aqui los callbacks que quieras aplicar.
        callbacks = CallbackList([
            stop_training_callback,
            text_progress_bar_callback,
        ])

        # Entrenamiento del agente
        agent.learn(total_timesteps=TRAIN_EPISODES * TIMESTEPS_PER_EPISODE,
                    callback=callbacks,
                    )

        # Evaluacion de la politica del agente
        episode_rewards = []
        for eval_seed in EVAL_SEEDS:
            eval_env.seed(eval_seed)
            rewards, _ = evaluate_policy(
                agent,
                eval_env,
                n_eval_episodes=1,
                deterministic=True,
                return_episode_rewards=True,
            )
            episode_rewards.extend(rewards)
        mean_reward = float(np.mean(episode_rewards))

        # Si el agente es el mejor hasta ahora, se guarda el agente junto con informacion relevante
        completed_trials = trial.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

        if not completed_trials or mean_reward > trial.study.best_value:
            agent.save(f"agentes/{EXPERIMENT_NAME}/best_model.zip")

            rewards = np.concatenate(train_env.env_method("get_episode_rewards"))
            lengths = np.concatenate(train_env.env_method("get_episode_lengths"))

            best_training_info.update({
                "rewards": rewards.copy(),
                "lengths": lengths.copy(),
                "trial_number": trial.number,
                "validation_mean_reward": mean_reward,
            })

        # Se devuelve la recompensa media obtenida durante la evaluacion del agente entrenado
        return mean_reward

    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()


if __name__ == "__main__":
    # Crear estudio de Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_COMBINACIONES // 3),
    )

    # Guardar la hora de inicio
    start_time = time.time()

    # Ejecutar entrenamiento
    study.optimize(objective, n_trials=N_COMBINACIONES, n_jobs=N_JOBS)

    # Guardar la hora de finalizacion
    end_time = time.time()

    # Mostrar informacion sobre el entrenamiento
    print(f"El entrenamiento {EXPERIMENT_NAME} ha durado {end_time - start_time:.2f} segundos.")
    print(f"Mejor recompensa obtenida: {study.best_value:.2f}")
    print(f"Se han probado {N_COMBINACIONES} combinaciones de hiperparametros.")
    print(f"Mejores hiperparametros:\n{study.best_params}")

    # Escribir informacion util sobre el entrenamiento en un archivo de texto
    with open(f"./agentes/{EXPERIMENT_NAME}/info_train.txt", "w") as f:
        f.write(f"Informacion sobre el experimento {EXPERIMENT_NAME}\n")
        f.write(f"\n - Duracion: {end_time - start_time:.2f} segundos.")
        f.write(f"\n - La mejor recompensa obtenida ha sido: {study.best_value:.2f}")
        f.write(f"\n - Se han probado {N_COMBINACIONES} combinaciones de hiperparametros.")
        f.write(f"\n - Los hiperparametros del mejor modelo encontrado son:\n\t{study.best_params}")

    # Evolucion de la recompensa durante el entrenamiento del mejor trial.
    WINDOW = 10  # Numero de episodios para la media movil
    training_rewards = best_training_info["rewards"]
    training_episodes = np.arange(1, len(training_rewards) + 1)

    # Calcular la media movil de la recompensa obtenida
    rolling_mean = np.array([
        training_rewards[max(0, i - WINDOW + 1):i + 1].mean()
        for i in range(len(training_rewards))
    ])

    # Graficar la evolucion de la recompensa durante el entrenamiento
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(training_episodes, training_rewards, alpha=0.35, label="Recompensa por episodio")
    ax.plot(training_episodes, rolling_mean, label=f"Media movil ({WINDOW} episodios)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa acumulada por episodio")
    ax.set_title(f"Mejor entrenamiento: trial {best_training_info['trial_number']}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'agentes/{EXPERIMENT_NAME}/reward_evolution.png')
    if args.show_plots:
        plt.show()
    plt.close(fig)