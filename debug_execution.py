import pandas as pd

# Read predictions để debug
test_date = '2020-04-06'
print(f"Debugging Bull market ({test_date}) improved strategy execution")
print("="*80)

# Load and run predictor
import sys
sys.path.append('.')
from predict_market_adaptive import MarketAdaptivePredictor

predictor = MarketAdaptivePredictor(
    'src/checkpoints_dqn/checkpoint_best.pkl',
    'data/raw/multi_coin_1h.csv'
)

test_df = predictor.load_data(test_date)
agent, env = predictor.load_model(test_df)
pred_df = predictor.get_raw_predictions(agent, test_df)
market_regime, _ = predictor.detect_market_regime(test_df)
pred_df, corrections = predictor.apply_market_adaptive_rules(pred_df, test_df, market_regime)

# Manual execution to debug
print("\nManual Execution Debug:")
print(f"{'Hour':<5} {'Action':<5} {'Price':<10} {'Pos Before':<12} {'Pos After':<12} {'Balance':<12} {'Holdings':<12}")
print("="*100)

balance = 10000.0
holdings = 0.0
position = 0
fee = 0.001

for idx, row in pred_df.iterrows():
    action = row['improved_action']
    action_name = row['improved_action_name']
    price = row['price']
    pos_before = position
    
    if action == 1 and position == 0:  # Buy
        holdings = balance / (price * (1 + fee))
        balance = 0
        position = 1
    elif action == 2 and position == 1:  # Sell
        balance = holdings * price * (1 - fee)
        holdings = 0
        position = 0
    
    portfolio = balance + holdings * price
    print(f"{idx:<5} {action_name:<5} ${price:<9.2f} {pos_before:<12} {position:<12} ${balance:<11.2f} {holdings:<12.4f} ${portfolio:.2f}")

print("\n" + "="*100)
print(f"Final Portfolio: ${portfolio:.2f}")
print(f"Return: {(portfolio - 10000) / 10000 * 100:.2f}%")
