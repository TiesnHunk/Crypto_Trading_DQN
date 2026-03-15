"""
Quick DQN+PSO+LSTM Training Script (CPU) - For Testing
Faster version with reduced episodes and particles for quick validation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.multi_coin_loader import MultiCoinLoader
from src.models.mdp_trading import TradingMDP
from src.models.dqn_pso_lstm_trading import DQNLSTMAgent, DQNPSOLSTMOptimizer

def load_data(coin='BTC'):
    """Load and prepare data"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load from existing file
    df = pd.read_csv('data/raw/multi_coin_1h.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"   Total rows: {len(df):,}")
    
    # Filter for specific coin
    df_coin = df[df['coin'] == coin].copy()
    print(f"   {coin} data: {len(df_coin):,} rows")
    print(f"   Date range: {df_coin['timestamp'].min()} to {df_coin['timestamp'].max()}")
    
    # Split train/test
    train_size = int(len(df_coin) * 0.8)
    train_df = df_coin.iloc[:train_size].copy()
    test_df = df_coin.iloc[train_size:].copy()
    
    print(f"\nData Split:")
    print(f"   Train: {len(train_df):,} samples ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"   Test: {len(test_df):,} samples ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    return train_df, test_df

def train_with_pso(train_df):
    """Quick PSO training with reduced parameters"""
    print("\n" + "="*80)
    print("QUICK PSO OPTIMIZATION")
    print("="*80)
    
    # Create environment
    env = TradingMDP(train_df)
    
    # Get input size
    feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'coin']]
    input_size = len(feature_cols)
    print(f"   Input size: {input_size} features")
    print(f"   Environment: {len(train_df):,} steps")
    
    # PSO Configuration - REDUCED for quick testing
    pso_config = {
        'n_particles': 2,      # Reduced from 3
        'max_iterations': 3,     # Reduced from 5
    }
    
    print(f"\n   PSO Configuration:")
    print(f"   - Particles: {pso_config['n_particles']}")
    print(f"   - Iterations: {pso_config['max_iterations']}")
    print(f"   - Device: cpu")
    
    # Create optimizer
    optimizer = DQNPSOLSTMOptimizer(**pso_config)
    
    # Run optimization
    print(f"\n{'='*80}")
    print("Starting Quick PSO Optimization...")
    print(f"{'='*80}\n")
    
    best_params, best_agent, history = optimizer.optimize(
        env=env,
        input_size=input_size,
        n_final_episodes=10,  # Reduced for quick testing
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print("PSO OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Best Fitness: {history['global_best_fitness'][-1]:.2f}")
    print(f"   Best Params: {best_params}")
    
    return best_params, best_agent, optimizer

def train_agent(agent, train_df, episodes=10):
    """Quick training with reduced episodes"""
    print("\n" + "="*80)
    print("QUICK TRAINING PHASE")
    print("="*80)
    print(f"   Episodes: {episodes}")
    print(f"   Batch size: {agent.batch_size}")
    
    env = TradingMDP(train_df)
    
    feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'coin']]
    env_data = train_df[feature_cols].values
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            state_seq = agent.prepare_sequence(env_data, env.current_step)
            action = agent.select_action(state_seq, training=True)
            next_state, reward, done, _ = env.step(action)
            next_state_seq = agent.prepare_sequence(env_data, env.current_step)
            
            agent.store_transition(state_seq, action, reward, next_state_seq, done)
            
            # Train when buffer has enough samples
            if len(agent.buffer) >= min(agent.batch_size, 200):
                agent.train_step()
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        agent.update_epsilon()
        rewards_history.append(episode_reward)
        
        print(f"   Ep {episode+1}/{episodes}: Reward={episode_reward:.2f}, Steps={steps}, Buffer={len(agent.buffer)}, Eps={agent.epsilon:.4f}")
    
    return rewards_history

def evaluate_agent(agent, test_df):
    """Quick evaluation"""
    print("\n" + "="*80)
    print("QUICK EVALUATION")
    print("="*80)
    
    env = TradingMDP(test_df)
    
    feature_cols = [col for col in test_df.columns if col not in ['timestamp', 'coin']]
    env_data = test_df[feature_cols].values
    
    state = env.reset()
    done = False
    total_reward = 0
    actions_taken = []
    
    while not done:
        state_seq = agent.prepare_sequence(env_data, env.current_step)
        action = agent.predict_action(state_seq)
        state, reward, done, _ = env.step(action)
        
        total_reward += reward
        actions_taken.append(action)
    
    print(f"\n   Test Results:")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Final Portfolio: ${env.balance:.2f}")
    print(f"   Total Return: {((env.balance - env.initial_balance) / env.initial_balance * 100):.2f}%")
    
    action_counts = pd.Series(actions_taken).value_counts()
    print(f"\n   Action Distribution:")
    print(f"   Hold: {action_counts.get(0, 0)} ({action_counts.get(0, 0)/len(actions_taken)*100:.1f}%)")
    print(f"   Buy: {action_counts.get(1, 0)} ({action_counts.get(1, 0)/len(actions_taken)*100:.1f}%)")
    print(f"   Sell: {action_counts.get(2, 0)} ({action_counts.get(2, 0)/len(actions_taken)*100:.1f}%)")
    
    return total_reward, env.balance

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("QUICK DQN + PSO + LSTM TRAINING (CPU)")
    print("="*80)
    print(f"\n   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Coin: BTC")
    print(f"   Mode: Quick Testing")
    
    # Set seed
    np.random.seed(42)
    
    # Load data
    train_df, test_df = load_data('BTC')
    
    # PSO Optimization (reduced)
    best_params, agent, optimizer = train_with_pso(train_df)
    
    # Quick Training (reduced episodes)
    print("\nStarting quick training...")
    rewards = train_agent(agent, train_df, episodes=10)
    
    # Evaluation
    test_reward, final_balance = evaluate_agent(agent, test_df)
    
    # Save checkpoint
    checkpoint_dir = "checkpoints/dqn_pso_lstm_quick"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "quick_test.pth")
    agent.save_checkpoint(checkpoint_path, best_params)
    print(f"\nCheckpoint saved: {checkpoint_path}")
    
    print("\n" + "="*80)
    print("QUICK TRAINING COMPLETE")
    print("="*80)
    print(f"\n   Best PSO Fitness: {optimizer.fitness_history[-1]:.2f}")
    print(f"   Test Reward: {test_reward:.2f}")
    print(f"   Final Balance: ${final_balance:.2f}")
    print(f"   Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
