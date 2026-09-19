import gymnasium as gym
import torch as th
 
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
 
# Custom MLP policy of two layers of size 32 each
class CustomPolicy_2x32(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=dict(
                pi=[32, 32],
                vf=[32, 32]
            )
        )
 
# Custom MLP policy of three layers of size 64 each
class CustomPolicy_3x64(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=dict(
                pi=[64, 64, 64],
                vf=[64, 64, 64]
            )
        )
 
# Custom MLP policy of two layers of size 64 each + a shared layer of size 64
class CustomPolicy_2x64_shared(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[64, dict(
                pi=[64, 64],
                vf=[64, 64]
            )]
        )
 
# Custom MLP policy of three layers of size 128 each
# + one shared layer of size 128
class CustomPolicy_4x128(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[128, dict(
                pi=[128, 128, 128],
                vf=[128, 128, 128]
            )]
        )
 
# Custom LSTM policy with two MLP layers of size 64 each + a shared LSTM layer of size 4
class CustomLSTMPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            lstm_hidden_size=4,
            net_arch=dict(
                pi=[64, 64],
                vf=[64, 64]
            ),
            enable_critic_lstm=True
        )
 
# Custom MLP policy of two layers of size 81 each
class CustomPolicy_2x81(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=dict(
                pi=[81, 81],
                vf=[81, 81]
            )
        )
 
# Custom MLP policy of three layers with variable size
class CustomPolicy_3_var(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=dict(
                pi=[70, 46, 30],
                vf=[70, 26, 10]
            )
        )
 
# NOTE: register_policy() does not exist in SB3.
# Pass the policy class directly when creating the model, e.g.:
#   model = PPO(CustomPolicy_2x32, env)
#   model = RecurrentPPO(CustomLSTMPolicy, env)