import gymnasium as gym
import gym_rendezvous

env = gym.make("rendezvous-v1")
obs, info = env.reset()

print("OK: entorno creado")
print("Observación:", obs)