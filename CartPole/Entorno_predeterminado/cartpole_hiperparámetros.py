import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import time

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# --------------------------------------------------------------------
# CONFIGURACIÓN DEL BARRIDO DE HIPERPARÁMETRO
# --------------------------------------------------------------------
valores_hiperparametro = [0.1, 0.5, 1, 5]  # valores a probar para el hiperparámetro
nombre_hiperparametro = "max_grad_norm"  # nombre del hiperparámetro que se está variando
SEMILLA = 42  # fija para que la única diferencia entre corridas sea el hiperparámetro

resultados = {}  # guarda df y métricas finales por cada valor

for valor in valores_hiperparametro:
    print(f"\n=== Entrenando con {nombre_hiperparametro} = {valor} ===")

    env = gym.make('CartPole-v1', render_mode="rgb_array")

    log_dir = f"./ppo_cartpole_logs/{nombre_hiperparametro}_{valor}/"
    os.makedirs(log_dir, exist_ok=True)

    model = PPO("MlpPolicy", env, verbose=0, max_grad_norm=valor, seed=SEMILLA)

    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)

    start_time = time.time()
    model.learn(total_timesteps=75000)
    end_time = time.time()

    df = pd.read_csv(os.path.join(log_dir, "progress.csv"))

    # Evaluación robusta (100 episodios, política determinista -> sin
    # ruido de exploración, ver comentario en la respuesta sobre por qué
    # esto puede dar 500 aunque la curva de entrenamiento aún oscile)
    eval_env = Monitor(gym.make('CartPole-v1'))
    episode_rewards, _ = evaluate_policy(
        model, eval_env, n_eval_episodes=100,
        deterministic=True, return_episode_rewards=True
    )
    episode_rewards = np.array(episode_rewards)
    tasa_exito = np.mean(episode_rewards >= 500) * 100
    eval_env.close()
    env.close()  # antes no se cerraba el env de entrenamiento

    resultados[valor] = {
        "df": df,
        "tiempo": end_time - start_time,
        "reward_media": episode_rewards.mean(),
        "reward_std": episode_rewards.std(),
        "tasa_exito": tasa_exito
    }

# --------------------------------------------------------------------
# GRÁFICA COMPARATIVA: curva de aprendizaje (todas las curvas juntas)
# --------------------------------------------------------------------
plt.figure(figsize=(9, 6))
for valor, data in resultados.items():
    df_reward = data["df"].dropna(subset=["rollout/ep_rew_mean"])
    plt.plot(df_reward["time/total_timesteps"], df_reward["rollout/ep_rew_mean"],
              label=f"{nombre_hiperparametro} = {valor}")

plt.xlabel("Total timesteps")
plt.ylabel("Recompensa media por episodio")
plt.title(f"Curvas de aprendizaje - PPO CartPole (barrido de {nombre_hiperparametro})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./ppo_cartpole_logs/comparacion_curvas_aprendizaje_{nombre_hiperparametro}.png")
plt.show()

# --------------------------------------------------------------------
# GRÁFICA COMPARATIVA: pérdidas (subplots, una curva por valor en cada subplot)
# --------------------------------------------------------------------
loss_cols = ["train/value_loss", "train/policy_gradient_loss", "train/entropy_loss"]
titles = ["Value Loss", "Policy Gradient Loss", "Entropy Loss"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for valor, data in resultados.items():
    df = data["df"]
    # Solo se recorren las columnas de pérdida que realmente existen en
    # este progress.csv, evitando un KeyError si alguna no se llegó a loggear
    cols_disponibles = [c for c in loss_cols if c in df.columns]
    df_losses = df.dropna(subset=cols_disponibles)
    for ax, col, title in zip(axes, loss_cols, titles):
        if col not in cols_disponibles:
            ax.set_title(f"{title} (no disponible)")
            continue
        ax.plot(df_losses["time/total_timesteps"], df_losses[col],
                 label=f"{nombre_hiperparametro} = {valor}")
        ax.set_xlabel("Total timesteps")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.savefig(f"./ppo_cartpole_logs/comparacion_curvas_perdida_{nombre_hiperparametro}.png")
plt.show()

# --------------------------------------------------------------------
# TABLA RESUMEN
# --------------------------------------------------------------------
print("\n=== RESUMEN COMPARATIVO ===")
for valor, data in resultados.items():
    print(f"{nombre_hiperparametro}={valor}: "
          f"reward_media={data['reward_media']:.2f} ± {data['reward_std']:.2f}, "
          f"tasa_exito={data['tasa_exito']:.2f}%, "
          f"tiempo={data['tiempo']:.1f}s")