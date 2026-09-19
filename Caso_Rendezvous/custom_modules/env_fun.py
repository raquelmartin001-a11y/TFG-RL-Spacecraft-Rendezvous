import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
 
# The multiprocessing implementation requires a function that
# can be called inside the process to instantiate a gymnasium env
def make_env(env_id, rank, seed=0, monitor=True, filename=None, **kwargs):
    """
    Utility function for multiprocessed env.
 
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for PRNG
    :param monitor: (bool) monitor the training process
    :param filename: (string) the location to save an optional log file
    :param kwargs: the kwargs to pass to the environment class
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.reset(seed=seed + rank)  # Gymnasium: seeding is done via reset()
        if monitor:
            env = Monitor(env, filename=filename)  # allow_early_resets removed in SB3
        return env
 
    set_random_seed(seed)  # SB3 replacement for set_global_seeds
    return _init