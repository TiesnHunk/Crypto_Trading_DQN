"""
Debug BTC Win Rate = 0% issue
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('src')
from models.dqn_agent import DQNAgent
import config.config as config

def prepare_state(data, index, state_dim=8):
    """Chuẩn bị state từ dữ liệu"""
    if index < 1:
        return None
    
    row = data.iloc[index]
    
    state = [
        row.get('close', 0),
        row.get('volume', 0), 
        row.get('rsi', 50),
        row.get('macd', 0),
        row.get('sma_20', row.get('close', 0)),
        row.get('price_change', 0),
        row.get('volatility', 0),
        row.get('adx', 25)
    ]
    
    state = np.array(state[:state_dim])
    state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return state

# Load data
print("Loading BTC data...")
data = pd.read_csv('data/raw/multi_coin_1h.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Filter BTC Oct-Dec 2024
btc_data = data[data['coin'] == 'BTC'].copy()
test_data = btc_data[(btc_data['timestamp'] >= '2024-10-01') & 
                     (btc_data['timestamp'] <= '2024-12-31')].reset_index(drop=True)

print(f"BTC test data: {len(test_data)} rows")
print(f"Date range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")

# Load checkpoint
checkpoint_path = Path('src/checkpoints_dqn/checkpoint_best.pkl')
if not checkpoint_path.exists():
    checkpoint_path = Path('checkpoints/dqn_pso_lstm/checkpoint_best.pkl')

if not checkpoint_path.exists():
    print("❌ No checkpoint found!")
    sys.exit(1)

print(f"\nLoading checkpoint: {checkpoint_path}")
import torch
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

state_dim = checkpoint.get('hyperparameters', {}).get('state_dim', 8)
action_dim = checkpoint.get('hyperparameters', {}).get('action_dim', 3)

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device='cpu')
agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])

# Simulate trading
print("\nSimulating BTC trading...")
balance = 10000.0
holdings = 0.0
trades = []
actions_log = []  # Track all actions
states_sample = []  # Sample states to debug

for i in range(10, len(test_data)):
    current_price = test_data.loc[i, 'close']
    state = prepare_state(test_data, i, state_dim=state_dim)
    
    if state is None:
        continue
    
    # Log first 5 states for debugging
    if len(states_sample) < 5:
        states_sample.append({'step': i, 'state': state.copy(), 'price': current_price, 'holdings': holdings, 'balance': balance})
    
    action = agent.select_action(state, epsilon=0)
    actions_log.append(action)  # Log every action
    
    # Action mapping theo MDP: 0=Buy, 1=Sell, 2=Hold
    if action == 0:  # Buy (Fixed)
        if balance > 0:
            amount = balance * 0.95
            can_buy = amount / current_price
            holdings += can_buy
            balance -= amount
            trades.append({
                'time': test_data.loc[i, 'timestamp'],
                'action': 'BUY',
                'price': current_price,
                'amount': can_buy
            })
    
    elif action == 1:  # Sell (Fixed)
        if holdings > 0:
            amount = holdings * 0.95
            balance += amount * current_price
            holdings -= amount
            trades.append({
                'time': test_data.loc[i, 'timestamp'],
                'action': 'SELL',
                'price': current_price,
                'amount': amount
            })

# Analyze trades
print(f"\nBTC Trading Analysis:")
print(f"   Total actions: {len(actions_log)}")
print(f"   Total trades: {len(trades)}")

# Action distribution
from collections import Counter
action_dist = Counter(actions_log)
print(f"\nAction Distribution (0=Buy, 1=Sell, 2=Hold):")
print(f"   Buy (0):  {action_dist.get(0, 0)} ({action_dist.get(0, 0)/len(actions_log)*100:.1f}%)")
print(f"   Sell (1): {action_dist.get(1, 0)} ({action_dist.get(1, 0)/len(actions_log)*100:.1f}%)")
print(f"   Hold (2): {action_dist.get(2, 0)} ({action_dist.get(2, 0)/len(actions_log)*100:.1f}%)")

# Show sample states
print(f"\nSample States (first 5):")
for i, s in enumerate(states_sample):
    print(f"   Step {s['step']}: price=${s['price']:.2f}, holdings={s['holdings']:.4f}, balance=${s['balance']:.2f}")
    print(f"      State: {s['state']}")
    print()

num_buy = len([t for t in trades if t['action'] == 'BUY'])
num_sell = len([t for t in trades if t['action'] == 'SELL'])

print(f"   BUY trades: {num_buy}")
print(f"   SELL trades: {num_sell}")

# Calculate win rate
print("\n🔍 Win Rate Calculation:")
winning_trades = 0
losing_trades = 0

for i, trade in enumerate(trades):
    if trade['action'] == 'SELL' and i > 0:
        # Find corresponding buy
        for j in range(i-1, -1, -1):
            if trades[j]['action'] == 'BUY':
                buy_price = trades[j]['price']
                sell_price = trade['price']
                profit = sell_price - buy_price
                
                if sell_price > buy_price:
                    winning_trades += 1
                    print(f"   ✅ SELL #{i}: Buy ${buy_price:.2f} → Sell ${sell_price:.2f} (profit: ${profit:.2f})")
                else:
                    losing_trades += 1
                    print(f"   ❌ SELL #{i}: Buy ${buy_price:.2f} → Sell ${sell_price:.2f} (loss: ${profit:.2f})")
                break

total_pairs = winning_trades + losing_trades
win_rate = (winning_trades / total_pairs * 100) if total_pairs > 0 else 0

print(f"\n📈 Results:")
print(f"   Winning trades: {winning_trades}")
print(f"   Losing trades: {losing_trades}")
print(f"   Total pairs: {total_pairs}")
print(f"   Win Rate: {win_rate:.1f}%")

# Show first few trades
print("\n📋 First 10 trades:")
for i, trade in enumerate(trades[:10]):
    print(f"   {i+1}. {trade['time'].strftime('%Y-%m-%d %H:%M')} - {trade['action']} at ${trade['price']:.2f}")

# Check if all trades are BUY only
if num_sell == 0:
    print("\n⚠️  WARNING: No SELL trades found! All trades are BUY.")
    print("   This explains why Win Rate = 0%")
    print("   The agent is only buying and never selling!")
