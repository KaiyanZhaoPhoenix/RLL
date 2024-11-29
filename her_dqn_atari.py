import os
import numpy as np
import random
import time
import wandb
import argparse
import torch
from envs.atari_env import create_atari_env
from algo.atari_her_dqn import AtariHerDQN

def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def initialize_wandb(args):
    """初始化WandB"""
    return wandb.init(
        project="gen",
        name=f"her_dqn_{args.env}_{args.seed}",
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
        notes="Training HER-DQN on Atari environment",
        mode='online'
    )

def main(args):
    # 初始化配置和种子
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    set_seed(args.seed)

    # 设置设备
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 初始化WandB
    run = initialize_wandb(args)

    # 创建环境
    env = create_atari_env(args.env)
    env.reset(seed=args.seed)

    # 创建模型
    model = AtariHerDQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        n_sampled_transitions=args.n_sampled_transitions,
        score_percentile=args.score_percentile,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=f"runs/her_dqn_{args.env}",
        device=device,
        verbose=1
    )

    # 训练模型
    model.learn(total_timesteps=args.total_timesteps)

    # 保存模型
    if args.save_model:
        model_path = os.path.join("models", f"her_dqn_{args.env}_{args.seed}")
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")

    # 结束WandB运行
    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Atari HER-DQN Training')

    # 环境参数
    parser.add_argument('--env', type=str, default="ALE/Breakout-v5", 
                       help='Atari environment name')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed')
    parser.add_argument('--total_timesteps', type=int, default=1000000, 
                       help='Total timesteps for training')

    # 模型参数
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=100000, 
                       help='Size of replay buffer')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--learning_starts', type=int, default=10000, 
                       help='Learning start timestep')
    parser.add_argument('--tau', type=float, default=1.0, 
                       help='Soft update coefficient')
    parser.add_argument('--gamma', type=float, default=0.99, 
                       help='Discount factor')
    parser.add_argument('--train_freq', type=int, default=4, 
                       help='Training frequency')
    parser.add_argument('--gradient_steps', type=int, default=1, 
                       help='Gradient steps')
    parser.add_argument('--target_update_interval', type=int, default=1000, 
                       help='Target network update interval')

    # HER特定参数
    parser.add_argument('--n_sampled_transitions', type=int, default=4, 
                       help='Number of transitions to sample')
    parser.add_argument('--score_percentile', type=float, default=75.0, 
                       help='Score percentile for selecting high-reward states')

    # 探索参数
    parser.add_argument('--exploration_fraction', type=float, default=0.1, 
                       help='Exploration fraction')
    parser.add_argument('--exploration_final_eps', type=float, default=0.02, 
                       help='Final exploration epsilon')

    # 其他参数
    parser.add_argument('--use_cuda', type=bool, default=True, 
                       help='Use CUDA if available')
    parser.add_argument('--save_model', type=bool, default=True, 
                       help='Save model after training')

    args = parser.parse_args()

    # 训练
    training_start_time = time.time()
    main(args)
    training_duration = time.time() - training_start_time
    print(f'Training time: {training_duration / 3600:.2f} hours') 