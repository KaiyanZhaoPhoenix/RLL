import os
import numpy as np
import random
import time
import wandb
import argparse
import torch
from envs.atari_env import create_atari_env
from algo.atari_per_dqn import PERDQN

def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def initialize_wandb(args):
    """初始化Weights & Biases用于实验追踪"""
    return wandb.init(
        project="gen",  # 指定你的WandB项目名
        name=f"{args.env}_PER_DQN_{args.seed}",  # 运行名称
        config=vars(args),  # 自动记录参数配置
        sync_tensorboard=True,  # 同步TensorBoard日志
        save_code=True,  # 保存代码
        notes="Training PER-DQN on Atari environment",  # 运行说明
        mode='online'  # 确保在线记录
    )

def main(args):
    # 初始化配置和种子
    env_id = args.env
    policy_type = args.policy_type
    total_timesteps = args.total_timesteps
    use_cuda = args.use_cuda

    # 如果没有指定种子,随机生成一个
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    set_seed(args.seed)

    # 设备设置(如果可用则使用CUDA,否则使用CPU)
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 初始化Weights & Biases进行实验追踪
    run = initialize_wandb(args)

    # 创建并重置Atari环境
    env = create_atari_env(env_id)
    env.reset(seed=args.seed)

    # 初始化PER-DQN模型
    model = PERDQN(
        policy=policy_type,
        env=env,
        learning_rate=2.5e-4,
        buffer_size=100000,  # 减小缓冲区大小
        learning_starts=10000,  # 减小学习开始步数
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        per_alpha=0.6,
        per_beta=0.4,
        per_eps=1e-6,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        max_grad_norm=10,
        tensorboard_log=f"runs/per_dqn_{env_id}",
        verbose=1,
    )

    # 训练模型
    model.learn(total_timesteps=total_timesteps)

    # 结束wandb运行并记录最终模型
    run.finish()

if __name__ == '__main__':
    # 参数解析器
    parser = argparse.ArgumentParser(description='Atari PER-DQN Training')

    # 通用配置
    parser.add_argument('--env', type=str, default="ALE/Breakout-v5", 
                       help='Atari环境名称')
    parser.add_argument('--policy_type', type=str, default="CnnPolicy", 
                       help='策略类型(通常为"CnnPolicy")')
    parser.add_argument('--total_timesteps', type=int, default=1000000, 
                       help='总训练步数')
    parser.add_argument('--seed', type=int, default=None, 
                       help='随机种子')
    parser.add_argument('--use_cuda', type=bool, default=True, 
                       help='是否使用CUDA训练')

    args = parser.parse_args()

    # 记录训练时间
    training_start_time = time.time()
    main(args)
    training_duration = time.time() - training_start_time
    print(f'Training time: {training_duration / 3600:.2f} hours')
