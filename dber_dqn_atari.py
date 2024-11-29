import numpy as np
import random
import time
import wandb
import argparse
import torch
from envs.atari_env import create_atari_env
from algo.atari_dber_dqn.dber_dqn import DBERDQN

def set_seed(seed: int):
    """设置随机种子，确保实验可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def initialize_wandb(args):
    """初始化W&B，用于记录实验数据和可视化"""
    run = wandb.init(
        project="gen",
        config=vars(args),
        name=f"dber_dqn_{args.env}_{args.seed}_{int(time.time())}",
        monitor_gym=True,
        save_code=True,
        sync_tensorboard=True,
    )
    return run

def main(args):
    """主训练循环"""
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    # 初始化wandb
    if args.use_wandb:
        run = initialize_wandb(args)
    
    # 创建环境
    env = create_atari_env(args.env)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using {device} device")
    
    # 创建模型
    model = DBERDQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=50000,
        learning_starts=10000,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        segment_length=args.segment_length,  # DBER特有参数
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log="./tensorboard_log" if args.use_wandb else None,
        verbose=1,
        device=device,
    )
    
    # 开始训练
    model.learn(
        total_timesteps=args.total_timesteps,
    )
    
    # 关闭环境
    env.close()
    
    # 关闭wandb
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DBER-DQN for Atari')
    
    # 环境参数
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5",
                       help="Atari environment name")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=10_000_000,
                       help="Total timesteps to train")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--buffer_size", type=int, default=500,
                       help="Size of the replay buffer")
    parser.add_argument("--learning_starts", type=int, default=10000,
                       help="How many steps before learning starts")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Minibatch size")
    parser.add_argument("--tau", type=float, default=1.0,
                       help="Soft update coefficient")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--train_freq", type=int, default=4,
                       help="Update the model every `train_freq` steps")
    parser.add_argument("--gradient_steps", type=int, default=1,
                       help="How many gradient steps to do after each rollout")
    parser.add_argument("--target_update_interval", type=int, default=10_000,
                       help="Update the target network every `target_update_interval` steps")
    
    # 探索参数
    parser.add_argument("--exploration_fraction", type=float, default=0.1,
                       help="Fraction of total timesteps for exploration")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0,
                       help="Initial exploration rate")
    parser.add_argument("--exploration_final_eps", type=float, default=0.01,
                       help="Final exploration rate")
    parser.add_argument("--max_grad_norm", type=float, default=10,
                       help="Maximum gradient norm for gradient clipping")
    
    # DBER特有参数
    parser.add_argument("--segment_length", type=int, default=2,
                       help="Length of trajectory segments")
    
    # 其他设置
    parser.add_argument("--use_cuda", type=bool, default=True,
                       help="Enable CUDA training if available")
    parser.add_argument("--use_wandb", type=bool, default=True,
                       help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    training_start_time = time.time()
    main(args)
    training_duration = time.time() - training_start_time
    print(f"训练完成，总用时: {training_duration / 3600:.2f} 小时") 