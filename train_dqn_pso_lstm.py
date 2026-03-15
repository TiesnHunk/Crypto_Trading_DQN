"""
Train DQN + PSO + LSTM từ đầu với checkpoint
Training script for DQN+PSO+LSTM model from scratch
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.dqn_pso_lstm_trading import DQNPSOLSTMOptimizer, DQNLSTMAgent
from src.models.mdp_trading import TradingMDP


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train DQN+PSO+LSTM model')
    
    # Data args
    parser.add_argument('--data', type=str, default='data/raw/multi_coin_1h.csv',
                       help='Path to data file')
    parser.add_argument('--coin', type=str, default='BTC',
                       help='Coin to train on (BTC, ETH, BNB, SOL, ADA)')
    
    # PSO args
    parser.add_argument('--particles', type=int, default=5,
                       help='Number of PSO particles')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of PSO iterations')
    
    # Training args
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Train/test split ratio')
    
    # Checkpoint args
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/dqn_pso_lstm',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=20,
                       help='Save checkpoint every N episodes')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')
    
    # Other args
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def load_data(data_path: str, coin: str = 'BTC'):
    """
    Load and prepare data
    """
    print("\n" + "="*80)
    print("📥 LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"   Total rows: {len(df):,}")
    
    # Filter coin
    df = df[df['coin'] == coin].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   {coin} data: {len(df):,} rows")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def split_data(df: pd.DataFrame, train_split: float = 0.8):
    """
    Split data into train and test
    """
    split_idx = int(len(df) * train_split)
    
    train_df = df[:split_idx].copy()
    test_df = df[split_idx:].copy()
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(train_df):,} samples ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"   Test: {len(test_df):,} samples ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    return train_df, test_df


def prepare_features(df):
    """
    Prepare features for model
    """
    features = ['open', 'high', 'low', 'close', 'volume',
               'rsi', 'macd', 'macd_signal', 'macd_hist',
               'sma_20', 'sma_50', 'ema_12', 'ema_26',
               'bb_upper', 'bb_middle', 'bb_lower',
               'adx', 'price_change', 'volatility']
    
    data = df[features].copy()
    data = data.fillna(method='ffill').fillna(0)
    
    return data.values


def train_with_pso(train_df, args):
    """
    Train model with PSO optimization
    """
    print("\n" + "="*80)
    print("🔬 PSO OPTIMIZATION PHASE")
    print("="*80)
    
    # Create environment
    env = TradingMDP(
        data=train_df,
        initial_balance=10000,
        transaction_cost=0.0001
    )
    
    # Get input size
    X_train = prepare_features(train_df)
    input_size = X_train.shape[1]
    
    print(f"   Input size: {input_size} features")
    print(f"   Environment: {len(train_df):,} steps")
    
    # Create optimizer
    optimizer = DQNPSOLSTMOptimizer(
        n_particles=args.particles,
        max_iterations=args.iterations,
        w=0.7, c1=1.5, c2=1.5
    )
    
    print(f"\n   PSO Configuration:")
    print(f"   - Particles: {args.particles}")
    print(f"   - Iterations: {args.iterations}")
    print(f"   - Inertia weight: 0.7")
    print(f"   - Cognitive/Social: 1.5/1.5")
    
    # Run PSO optimization
    print(f"\n{'='*80}")
    print("🚀 Starting PSO Optimization...")
    print('='*80)
    
    best_params, best_agent, history = optimizer.optimize(
        env=env,
        input_size=input_size,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print("✅ PSO OPTIMIZATION COMPLETE!")
    print('='*80)
    print(f"\n   Best Parameters:")
    for key, value in best_params.items():
        print(f"      {key}: {value}")
    print(f"\n   Best Fitness: {history['global_best_fitness'][-1]:.2f}")
    
    # Save PSO results
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    pso_results_path = checkpoint_dir / 'pso_optimization_results.json'
    optimizer.save_results(best_agent, best_params, str(pso_results_path))
    
    return best_params, best_agent, optimizer


def train_agent(agent, train_df, test_df, args, checkpoint_dir):
    """
    Train agent with best parameters
    """
    print("\n" + "="*80)
    print("🎯 TRAINING PHASE")
    print("="*80)
    
    # Create training environment
    train_env = TradingMDP(
        data=train_df,
        initial_balance=10000,
        transaction_cost=0.0001
    )
    
    # Prepare features
    train_data = prepare_features(train_df)
    
    # Training stats
    episode_rewards = []
    best_reward = -float('inf')
    best_test_reward = -float('inf')
    
    print(f"\n   Training Configuration:")
    print(f"   - Episodes: {args.episodes}")
    print(f"   - Sequence length: {agent.sequence_length}")
    print(f"   - Batch size: {agent.batch_size}")
    print(f"   - Learning rate: {agent.learning_rate}")
    print(f"   - Gamma: {agent.gamma}")
    print(f"   - Epsilon decay: {agent.epsilon_decay}")
    
    print(f"\n{'='*80}")
    print("🚀 Starting Training...")
    print('='*80)
    
    for episode in range(args.episodes):
        # Reset environment
        state = train_env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        done = False
        
        while not done:
            # Get state sequence
            state_seq = agent.prepare_sequence(train_data, train_env.current_step)
            
            # Select action
            action = agent.select_action(state_seq)
            
            # Take step
            next_state, reward, done, info = train_env.step(action)
            
            # Get next state sequence
            next_state_seq = agent.prepare_sequence(train_data, train_env.current_step)
            
            # Store transition
            agent.store_transition(state_seq, action, reward, next_state_seq, done)
            
            # Train
            if len(agent.buffer) >= agent.batch_size:
                loss = agent.train_step()
                episode_loss += loss
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        # Update epsilon
        agent.update_epsilon()
        
        # Record stats
        episode_rewards.append(episode_reward)
        avg_loss = episode_loss / steps if steps > 0 else 0
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            
            # Save best checkpoint
            best_path = checkpoint_dir / 'checkpoint_best.pth'
            agent.save_checkpoint(str(best_path), episode=episode+1, best_reward=best_reward)
        
        # Evaluate on test set periodically
        if (episode + 1) % 10 == 0:
            test_reward = evaluate_agent(agent, test_df)
            
            if test_reward > best_test_reward:
                best_test_reward = test_reward
                test_path = checkpoint_dir / 'checkpoint_best_test.pth'
                agent.save_checkpoint(str(test_path), episode=episode+1, best_reward=test_reward)
        else:
            test_reward = None
        
        # Print progress
        if (episode + 1) % 1 == 0:
            avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            
            print(f"Episode {episode+1}/{args.episodes}:")
            print(f"   Reward: {episode_reward:.2f} | Avg(10): {avg_reward_10:.2f} | Best: {best_reward:.2f}")
            print(f"   Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f} | Steps: {steps}")
            if test_reward is not None:
                print(f"   Test Reward: {test_reward:.2f} | Best Test: {best_test_reward:.2f}")
        
        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_ep{episode+1}.pth'
            agent.save_checkpoint(str(checkpoint_path), episode=episode+1, best_reward=best_reward)
            
            # Also save latest
            latest_path = checkpoint_dir / 'checkpoint_latest.pth'
            agent.save_checkpoint(str(latest_path), episode=episode+1, best_reward=best_reward)
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print('='*80)
    print(f"\n   Best Training Reward: {best_reward:.2f}")
    print(f"   Best Test Reward: {best_test_reward:.2f}")
    print(f"   Final Epsilon: {agent.epsilon:.4f}")
    
    # Save final checkpoint
    final_path = checkpoint_dir / 'checkpoint_final.pth'
    agent.save_checkpoint(str(final_path), episode=args.episodes, best_reward=best_reward)
    
    return episode_rewards


def evaluate_agent(agent, test_df):
    """
    Evaluate agent on test set
    """
    test_env = TradingMDP(
        data=test_df,
        initial_balance=10000,
        transaction_cost=0.0001
    )
    
    test_data = prepare_features(test_df)
    
    state = test_env.reset()
    total_reward = 0
    done = False
    
    while not done:
        state_seq = agent.prepare_sequence(test_data, test_env.current_step)
        action = agent.predict_action(state_seq)  # Greedy
        next_state, reward, done, _ = test_env.step(action)
        total_reward += reward
        state = next_state
    
    return total_reward


def plot_training_results(rewards, checkpoint_dir):
    """
    Plot training results
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, label='Episode Reward')
    
    # Moving average
    window = 10
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'MA({window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN+PSO+LSTM Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative reward
    plt.subplot(1, 2, 2)
    cumulative = np.cumsum(rewards)
    plt.plot(cumulative, 'g-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_path = checkpoint_dir / 'training_progress.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n   📊 Training plot saved: {plot_path}")
    
    plt.show()


def main():
    """
    Main function
    """
    args = parse_args()
    
    print("\n" + "="*80)
    print("🚀 DQN + PSO + LSTM TRAINING FROM SCRATCH")
    print("="*80)
    print(f"\n   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Coin: {args.coin}")
    print(f"   Device: {args.device or 'auto'}")
    print(f"   Seed: {args.seed}")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Checkpoint dir: {checkpoint_dir}")
    
    # Load data
    df = load_data(args.data, args.coin)
    
    # Split data
    train_df, test_df = split_data(df, args.train_split)
    
    # Check if resuming
    if args.resume:
        print("\n" + "="*80)
        print("📂 RESUMING FROM CHECKPOINT")
        print("="*80)
        
        # Load PSO results
        pso_results_path = Path(args.resume)
        
        optimizer = DQNPSOLSTMOptimizer(
            n_particles=args.particles,
            max_iterations=args.iterations
        )
        
        X_train = prepare_features(train_df)
        best_params, agent, history = optimizer.load_results(str(pso_results_path), X_train.shape[1])
        
        print(f"   Resumed from: {pso_results_path}")
    else:
        # PSO optimization
        best_params, agent, optimizer = train_with_pso(train_df, args)
    
    # Training
    rewards = train_agent(agent, train_df, test_df, args, checkpoint_dir)
    
    # Plot results
    plot_training_results(rewards, checkpoint_dir)
    
    # Final evaluation
    print("\n" + "="*80)
    print("📊 FINAL EVALUATION")
    print("="*80)
    
    final_test_reward = evaluate_agent(agent, test_df)
    print(f"\n   Final Test Reward: {final_test_reward:.2f}")
    
    # Calculate returns
    test_env = TradingMDP(data=test_df, initial_balance=10000, transaction_cost=0.0001)
    portfolio_value = test_env.initial_balance + final_test_reward
    total_return = (portfolio_value / test_env.initial_balance - 1) * 100
    
    print(f"   Initial Balance: ${test_env.initial_balance:.2f}")
    print(f"   Final Portfolio: ${portfolio_value:.2f}")
    print(f"   Total Return: {total_return:.2f}%")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n   All checkpoints saved to: {checkpoint_dir}")
    print(f"   - checkpoint_best.pth (best training reward)")
    print(f"   - checkpoint_best_test.pth (best test reward)")
    print(f"   - checkpoint_final.pth (final model)")
    print(f"   - checkpoint_latest.pth (latest model)")
    

if __name__ == "__main__":
    main()
