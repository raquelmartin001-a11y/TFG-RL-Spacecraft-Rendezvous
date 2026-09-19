import gymnasium as gym
import numpy as np
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import time


class CartPole(Env):
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.steps = 0
        self.max_steps = 500

        self.theta_threshold = 12 * np.pi / 180
        self.x_threshold = 2.4
        self.action_space = gym.spaces.Discrete(2)

        self.observation_space = gym.spaces.Box(
            low=np.array([-self.x_threshold, -np.inf, -self.theta_threshold, -np.inf], dtype=np.float32),
            high=np.array([self.x_threshold, np.inf, self.theta_threshold, np.inf], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        # CORRECCIÓN: estado inicial aleatorio uniforme, igual que CartPole-v1 oficial
        low = -0.05
        high = 0.05
        if options is not None:
            low = options.get("low", low)
            high = options.get("high", high)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        force = self.force_mag if action == 1 else -self.force_mag
        x, x_dot, theta, theta_dot = self.state

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        terminated = bool(
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold or theta > self.theta_threshold
        )

        truncated = self.steps >= self.max_steps
        reward = 1.0
        info = {}
        return self.state, reward, terminated, truncated, info

    def render(self):
        x, _, theta, _ = self.state
        cart_width = 0.4
        cart_height = 0.2
        pole_x = x + self.length * np.sin(theta)
        pole_y = cart_height / 2 + self.length * np.cos(theta)

        plt.clf()
        plt.xlim(-2.4, 2.4)
        plt.ylim(-1.0, 1.0)
        plt.gca().set_aspect('equal')
        plt.gca().add_patch(plt.Rectangle((x - cart_width/2, 0), cart_width, cart_height, color="black"))
        plt.plot([x, pole_x], [cart_height/2, pole_y], color="red", linewidth=3)
        plt.pause(0.001)


# --------------------------------------------------------------------
# 1. ENTRENAMIENTO CON LOGGING A CSV
# --------------------------------------------------------------------
env = CartPole()

log_dir = "./ppo_cartpole_mano_logs/"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir, exist_ok=True)

model = PPO("MlpPolicy", env, verbose=0)
new_logger = configure(log_dir, ["stdout", "csv"])
model.set_logger(new_logger)

start_time = time.time()
model.learn(total_timesteps=75000)
end_time = time.time()
model.save("ppo_cartpole_mano")

print(f"Timesteps totales entrenados: {model.num_timesteps}")

# --------------------------------------------------------------------
# 2. CURVA DE APRENDIZAJE Y CURVAS DE PÉRDIDA
# --------------------------------------------------------------------
df = pd.read_csv(os.path.join(log_dir, "progress.csv"))

if "rollout/ep_rew_mean" in df.columns:
    df_reward = df.dropna(subset=["rollout/ep_rew_mean"])
    plt.figure(figsize=(8, 5))
    plt.plot(df_reward["time/total_timesteps"], df_reward["rollout/ep_rew_mean"], marker='o')
    plt.xlabel("Total timesteps")
    plt.ylabel("Recompensa media por episodio (ep_rew_mean)")
    plt.title("Curva de aprendizaje - PPO CartPole (entorno propio)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "curva_aprendizaje.png"))
    plt.show()

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
# 3. VISUALIZACIÓN DE 10 EPISODIOS
# --------------------------------------------------------------------
episodes = 10
recompensa_total = 0
episod_exito = 0
t_total = 0

for episode in range(1, episodes + 1):
    obs, _ = env.reset()
    env.render()
    terminated = False
    truncated = False
    score = 0
    ep_start = time.time()
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()
    ep_end = time.time()
    if score >= 475:
        episod_exito += 1
    print(f"Episode: {episode}, Score: {score}")
    recompensa_total += score
    t_episodio = ep_end - ep_start
    print(f"Tiempo de simulación: {t_episodio:.2f} segundos")
    t_total += t_episodio

env.close()
print("---------------------------------------------------")
print(f"Recompensa total media en {episodes} episodios: {recompensa_total/episodes}")
print(f"Episodios exitosos (score>=475): {episod_exito/episodes*100} %")
print(f"Tiempo medio de simulación en {episodes} episodios: {t_total/episodes} segundos")
print("---------------------------------------------------")

# --------------------------------------------------------------------
# 4. EVALUACIÓN ROBUSTA: 100 episodios + histograma
# --------------------------------------------------------------------
eval_env = Monitor(CartPole())

episode_rewards, episode_lengths = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=100,
    deterministic=True,
    return_episode_rewards=True
)

episode_rewards = np.array(episode_rewards)
tasa_exito = np.mean(episode_rewards >= 500) * 100

print("---------------------------------------------------")
print(f"Evaluación sobre 100 episodios (deterministic=True):")
print(f"Recompensa media: {episode_rewards.mean():.2f}")
print(f"Desviación típica: {episode_rewards.std():.2f}")
print(f"Tasa de éxito (score >= 500): {tasa_exito:.2f} %")
print(f"Tiempo de simulación: {end_time-start_time:.2f} segundos")
print("---------------------------------------------------")

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