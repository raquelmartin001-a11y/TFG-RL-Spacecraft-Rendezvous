# Ejemplo del cartpole invertido usando aprendizaje por refuerzo
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
# 1. ENTRENAMIENTO CON LOGGING A CSV (para curvas de aprendizaje)
# --------------------------------------------------------------------
env = gym.make('CartPole-v1', render_mode="rgb_array")
env.reset(seed=123, options={"low": -0.1, "high": 0.1})

log_dir = "./ppo_cartpole_logs/"
os.makedirs(log_dir, exist_ok=True)

model = PPO("MlpPolicy", env, verbose=0, gamma=0.99)

# Configuramos el logger de SB3 para que escriba un progress.csv
new_logger = configure(log_dir, ["stdout", "csv"])
model.set_logger(new_logger)

start_time = time.time()
model.learn(total_timesteps=75000)
end_time = time.time()

# --------------------------------------------------------------------
# 2. LECTURA DEL CSV Y GRÁFICAS DE ENTRENAMIENTO
# --------------------------------------------------------------------
progress_path = os.path.join(log_dir, "progress.csv")
df = pd.read_csv(progress_path)

# --- Curva de aprendizaje: recompensa media por episodio vs timesteps ---
if "rollout/ep_rew_mean" in df.columns:
    df_reward = df.dropna(subset=["rollout/ep_rew_mean"])
    plt.figure(figsize=(8, 5))
    plt.plot(df_reward["time/total_timesteps"], df_reward["rollout/ep_rew_mean"], marker='o')
    plt.xlabel("Total timesteps")
    plt.ylabel("Recompensa media por episodio (ep_rew_mean)")
    plt.title("Curva de aprendizaje - PPO CartPole")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "curva_aprendizaje.png"))
    plt.show()

# --- Curvas de pérdida: value_loss, policy_gradient_loss, entropy_loss ---
loss_cols = ["train/value_loss", "train/policy_gradient_loss", "train/entropy_loss"]
df_losses = df.dropna(subset=[c for c in loss_cols if c in df.columns])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
titles = ["Value Loss", "Policy Gradient Loss", "Entropy Loss"]

for ax, col, title in zip(axes, loss_cols, titles):
    ax.plot(df_losses["time/total_timesteps"], df_losses[col], color='tab:blue')
    ax.set_xlabel("Total timesteps")
    ax.set_ylabel(col)
    ax.set_title(title)
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(log_dir, "curvas_perdida.png"))
plt.show()

# --------------------------------------------------------------------
# 3. EVALUACIÓN VISUAL (10 episodios)
# --------------------------------------------------------------------
#env_render = gym.make('CartPole-v1', render_mode="human")

#episodes = 10
#recompensa_total = 0
#episod_exito = 0
#t_total = 0

#for episode in range(1, episodes + 1):
#    obs, _ = env_render.reset()
#    env_render.render()
#    terminated = False
#    truncated = False
#    score = 0
#    while not (terminated or truncated):
#        action, _ = model.predict(obs, deterministic=True)
#        obs, reward, terminated, truncated, info = env_render.step(action)
#        score += reward
#        env_render.render()
#    if score >= 475:
#        episod_exito += 1
#    print(f"Episode: {episode}, Score: {score}")
#    recompensa_total += score

#env_render.close()
#print("---------------------------------------------------")
#print(f"Recompensa total media en {episodes} episodios: {recompensa_total/episodes}")
#print(f"Episodios exitosos (score>=475): {episod_exito/episodes*100} %")
#print(f"Tiempo total de simulación: {end_time - start_time:.2f} segundos")
#print("---------------------------------------------------")

# --------------------------------------------------------------------
# 4. EVALUACIÓN ROBUSTA: 100 episodios con evaluate_policy + histograma
# --------------------------------------------------------------------
eval_env = Monitor(gym.make('CartPole-v1'))

episode_rewards, episode_lengths = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=100,
    deterministic=True,
    return_episode_rewards=True
)

episode_rewards = np.array(episode_rewards)

# Tasa de éxito: episodios que alcanzan el score máximo (500)
tasa_exito = np.mean(episode_rewards >= 500) * 100

print("---------------------------------------------------")
print(f"Evaluación sobre 100 episodios (deterministic=True):")
print(f"Recompensa media: {episode_rewards.mean():.2f}")
print(f"Desviación típica: {episode_rewards.std():.2f}")
print(f"Tasa de éxito (score >= 500): {tasa_exito:.2f} %")
print(f"Tiempo total de simulación: {end_time - start_time:.2f} segundos")
print("---------------------------------------------------")

# Histograma de la distribución de recompensas
plt.figure(figsize=(8, 5))
plt.hist(episode_rewards, bins=20, edgecolor='black', color='tab:blue')
plt.xlabel("Score por episodio")
plt.ylabel("Número de episodios")
plt.title(f"Distribución de recompensas (100 episodios) - Éxito: {tasa_exito:.1f}%")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "histograma_evaluacion.png"))
plt.show()

eval_env.close()