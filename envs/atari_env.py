import gymnasium as gym
from ale_py import ALEInterface

def create_atari_env(env_id: str) -> gym.Env:
    ale = ALEInterface()
    gym.register_envs(ale)
    # Create env
    env = gym.make(env_id)
    
    return env
