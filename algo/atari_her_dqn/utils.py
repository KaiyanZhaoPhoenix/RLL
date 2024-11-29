import numpy as np
from typing import Dict, Any

def compute_mean_reward(rewards: np.ndarray) -> float:
    """计算平均奖励"""
    return float(np.mean(rewards))

def get_stats(episode_rewards: list) -> Dict[str, Any]:
    """
    计算episode统计信息
    """
    if len(episode_rewards) == 0:
        return {
            "mean_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
            "std_reward": 0.0,
        }
        
    rewards = np.array(episode_rewards)
    return {
        "mean_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "std_reward": float(np.std(rewards)),
    } 