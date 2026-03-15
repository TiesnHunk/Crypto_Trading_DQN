"""Debug Bull market predictions"""
import pandas as pd
import numpy as np
import torch
import sys
sys.path.append('src')

from models.dqn_agent import DQNAgent
from models.mdp_trading import TradingMDP

# Load data
df = pd.read_csv('data/raw/multi_coin_1h.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_btc = df[df['coin'] == 'BTC'].copy()
df_btc = df_btc.sort_values('timestamp').reset_index(drop=True)

target_date = pd.to_datetime('2020-04-06')
day_mask = df_btc['timestamp'].dt.date == target_date.date()
test_df = df_btc[day_mask].copy().reset_index(drop=True).iloc[:24]

# Load model
env = TradingMDP(test_df, initial_balance=10000.0)
agent = DQNAgent(state_dim=8, action_dim=3, learning_rate=0.001, gamma=0.99,
                 epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0,
                 buffer_capacity=10000, batch_size=64)

checkpoint = torch.load('src/checkpoints_dqn/checkpoint_best.pkl', map_location='cpu', weights_only=False)
agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
agent.policy_net.eval()
agent.device = 'cpu'

# Get predictions
state = env.reset()
print(f"{'Hour':<5} {'Time':<8} {'Price':<10} {'Trend':<7} {'RSI':<7} {'Pos':<4} {'Raw':<5} | Rule B2 Check")
print("="*90)

action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

for hour in range(24):
    timestamp = test_df.iloc[env.current_step]['timestamp'].strftime('%H:%M')
    price = test_df.iloc[env.current_step]['close']
    trend = test_df.iloc[env.current_step]['trend']
    rsi = test_df.iloc[env.current_step]['rsi']
    position = env.position
    
    action = agent.select_action(state, epsilon=0.0)
    action_name = action_names[action]
    
    # Check Rule B2 conditions
    b2a_check = f"trend>{trend:.2f}>0.3:{trend>0.3}, RSI<{rsi:.1f}<60:{rsi<60}, pos={position}==0:{position==0}"
    b2b_check = f"trend>{trend:.2f}>0.1:{trend>0.1}, RSI<{rsi:.1f}<45:{rsi<45}, pos={position}==0:{position==0}"
    
    if action == 0 and position == 0:  # Hold and no position
        if trend > 0.3 and rsi < 60:
            rule_result = "✓ B2a SHOULD BUY"
        elif trend > 0.1 and rsi < 45:
            rule_result = "✓ B2b SHOULD BUY"
        else:
            rule_result = f"✗ No match (trend={trend:.2f}, rsi={rsi:.1f})"
    else:
        rule_result = f"N/A (action={action_name}, pos={position})"
    
    print(f"{hour:<5} {timestamp:<8} ${price:<9.2f} {trend:>6.2f} {rsi:>6.1f} {position:<4} {action_name:<5} | {rule_result}")
    
    next_state, reward, done, info = env.step(action)
    state = next_state
    
    if done:
        break

print("\n" + "="*90)
print("SUMMARY:")
print(f"  - Hold actions with position=0: Check if RSI < 60 and trend > 0.3")
print(f"  - If RSI too high (>60), Rule B2 won't trigger")
print(f"  - Bull market avg RSI: 72.8 (too overbought for buying)")
