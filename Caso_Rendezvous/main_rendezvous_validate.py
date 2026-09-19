# CHECK THE ENVIRONMENT #

# CAMBIO 1: 'gym' reemplazado por 'gymnasium'
# gymnasium es el sucesor mantenido por la Farama Foundation. 
# El paquete gym de OpenAI quedó deprecado en 2023.
import gymnasium as gym
import gym_rendezvous

# CAMBIO 2: el checker ahora viene de stable_baselines3
# En SB (v1) estaba en stable_baselines.common.env_checker.
# En SB3 se encuentra en stable_baselines3.common.env_checker.
from stable_baselines3.common.env_checker import check_env

env = gym.make('rendezvous-v1')

# Si el entorno no sigue la interfaz de Gymnasium, se lanzará un error.
check_env(env, warn=True)
