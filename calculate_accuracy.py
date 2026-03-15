"""
Tính accuracy cho validation results
Accuracy = % dự đoán đúng xu hướng giá (tăng/giảm)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.dqn_agent import DQNAgent
from models.mdp_trading import TradingMDP

def calculate_accuracy(checkpoint_path="src/checkpoints_dqn/checkpoint_best.pkl",
                      data_path="data/raw/multi_coin_1h.csv"):
    """
    Tính accuracy theo cách của paper:
    - BUY action khi giá sẽ tăng trong N giờ tới = correct
    - SELL action khi giá sẽ giảm trong N giờ tới = correct
    - HOLD action = neutral (không tính vào accuracy)
    """
    
    print("="*80)
    print("CALCULATING TRADING ACCURACY")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_path}\n")
    
    # Load data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Validation period
    val_df = df[(df['timestamp'] >= '2023-08-01') & 
                (df['timestamp'] <= '2024-02-29')]
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    hyperparams = checkpoint.get('hyperparameters', {})
    state_dim = hyperparams.get('state_dim', 8)
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=3,
        device='cpu'
    )
    
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.epsilon = 0.0  # No exploration for accuracy test
    
    print(f"✅ Loaded Episode {checkpoint.get('episode', 0)}\n")
    
    # Calculate accuracy for each coin
    results = {}
    
    for coin in val_df['coin'].unique():
        coin_df = val_df[val_df['coin'] == coin].copy().reset_index(drop=True)
        
        print(f"📊 {coin}:")
        print(f"   Samples: {len(coin_df):,}")
        
        # Create environment
        mdp = TradingMDP(
            coin_df,
            initial_balance=10000.0,
            transaction_cost=0.0001,
            interval='1h',
            enable_risk_management=False
        )
        
        state = mdp.reset()
        
        # Track predictions
        correct_predictions = 0
        total_predictions = 0
        
        buy_correct = 0
        buy_total = 0
        sell_correct = 0
        sell_total = 0
        
        # Run through data
        for i in range(len(coin_df) - 24):  # -24 để có thể look ahead 24h
            # Get action from agent
            action = agent.select_action(state)
            
            # Get current and future prices
            current_price = coin_df.iloc[i]['close']
            
            # Look ahead 24 hours
            future_price = coin_df.iloc[i + 24]['close']
            price_change = (future_price - current_price) / current_price
            
            # Determine if prediction was correct
            if action == 0:  # BUY
                buy_total += 1
                if price_change > 0.01:  # Giá tăng > 1%
                    correct_predictions += 1
                    buy_correct += 1
                total_predictions += 1
                
            elif action == 1:  # SELL
                sell_total += 1
                if price_change < -0.01:  # Giá giảm > 1%
                    correct_predictions += 1
                    sell_correct += 1
                total_predictions += 1
            
            # Step to next state
            next_state, _, done, _ = mdp.step(action)
            state = next_state
            
            if done:
                break
        
        # Calculate accuracy
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        buy_accuracy = (buy_correct / buy_total * 100) if buy_total > 0 else 0
        sell_accuracy = (sell_correct / sell_total * 100) if sell_total > 0 else 0
        
        results[coin] = {
            'accuracy': accuracy,
            'buy_accuracy': buy_accuracy,
            'sell_accuracy': sell_accuracy,
            'total_predictions': total_predictions,
            'buy_total': buy_total,
            'sell_total': sell_total
        }
        
        print(f"   Overall Accuracy: {accuracy:.2f}%")
        print(f"   BUY Accuracy: {buy_accuracy:.2f}% ({buy_correct}/{buy_total})")
        print(f"   SELL Accuracy: {sell_accuracy:.2f}% ({sell_correct}/{sell_total})")
        print()
    
    # Calculate average
    avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
    avg_buy = np.mean([r['buy_accuracy'] for r in results.values()])
    avg_sell = np.mean([r['sell_accuracy'] for r in results.values()])
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Average BUY Accuracy: {avg_buy:.2f}%")
    print(f"Average SELL Accuracy: {avg_sell:.2f}%")
    print()
    
    print("📊 DETAILED RESULTS:")
    print(f"{'Coin':<8} {'Accuracy':<12} {'BUY Acc':<12} {'SELL Acc':<12} {'Predictions'}")
    print("-" * 70)
    for coin, r in results.items():
        print(f"{coin:<8} {r['accuracy']:>10.2f}%  {r['buy_accuracy']:>10.2f}%  "
              f"{r['sell_accuracy']:>10.2f}%  {r['total_predictions']:>6d}")
    print("-" * 70)
    print(f"{'AVERAGE':<8} {avg_accuracy:>10.2f}%  {avg_buy:>10.2f}%  "
          f"{avg_sell:>10.2f}%")
    print("="*80)
    
    # Compare with paper
    print("\n📄 COMPARISON WITH PAPER:")
    print("-" * 70)
    print("Paper - Dogecoin (Trend-based): 88.76%")
    print("Paper - Bitcoin (Trend-based):  55.29%")
    print(f"Our Model - Average:            {avg_accuracy:.2f}%")
    
    if avg_accuracy >= 55:
        print("✅ Our model's accuracy is competitive with paper!")
    else:
        print("⚠️  Lower accuracy than paper benchmarks")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', 
                       default='src/checkpoints_dqn/checkpoint_best.pkl',
                       help='Checkpoint file path')
    parser.add_argument('--data',
                       default='data/raw/multi_coin_1h.csv',
                       help='Data file path')
    
    args = parser.parse_args()
    
    results = calculate_accuracy(args.checkpoint, args.data)
