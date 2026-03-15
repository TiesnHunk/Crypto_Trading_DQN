"""
MARKET-ADAPTIVE IMPROVED PREDICTION
Sử dụng rules khác nhau tùy theo loại thị trường (Bull/Bear/Sideways)
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os
sys.path.append('src')

from models.dqn_agent import DQNAgent
from models.mdp_trading import TradingMDP

class MarketAdaptivePredictor:
    """Cải thiện predictions với market-adaptive rules"""
    
    def __init__(self, checkpoint_path, data_path):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        
    def load_data(self, date_str=None):
        """Load data và chọn ngày test"""
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_btc = df[df['coin'] == 'BTC'].copy()
        df_btc = df_btc.sort_values('timestamp').reset_index(drop=True)
        
        if date_str is None:
            end_idx = len(df_btc)
            start_idx = max(0, end_idx - 24)
            test_df = df_btc.iloc[start_idx:end_idx].copy()
        else:
            target_date = pd.to_datetime(date_str)
            day_mask = df_btc['timestamp'].dt.date == target_date.date()
            test_df = df_btc[day_mask].copy()
            if len(test_df) == 0:
                end_idx = len(df_btc)
                start_idx = max(0, end_idx - 24)
                test_df = df_btc.iloc[start_idx:end_idx].copy()
        
        test_df = test_df.iloc[:24].reset_index(drop=True)
        return test_df
    
    def load_model(self, test_df):
        """Load DQN model"""
        env = TradingMDP(test_df, initial_balance=10000.0)
        agent = DQNAgent(
            state_dim=8,
            action_dim=3,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_capacity=10000,
            batch_size=64
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.policy_net.eval()
        agent.policy_net.to('cpu')
        agent.target_net.to('cpu')
        agent.device = 'cpu'
        
        return agent, env
    
    def get_raw_predictions(self, agent, test_df):
        """Lấy predictions từ model gốc"""
        env = TradingMDP(test_df, initial_balance=10000.0)
        state = env.reset()
        
        predictions = []
        done = False
        hour = 0
        
        while not done and hour < 24:
            current_price = test_df.iloc[env.current_step]['close']
            timestamp = test_df.iloc[env.current_step]['timestamp']
            
            # Save position BEFORE action
            position_before = env.position
            balance_before = env.balance
            holdings_before = env.holdings
            
            action = agent.select_action(state, epsilon=0.0)
            action_name = self.action_names[action]
            
            next_state, reward, done, info = env.step(action)
            
            predictions.append({
                'hour': hour,
                'timestamp': timestamp,
                'raw_action': action,
                'raw_action_name': action_name,
                'state': state.copy(),
                'price': current_price,
                'reward': reward,
                'balance': balance_before,
                'holdings': holdings_before,
                'position': position_before
            })
            
            state = next_state
            hour += 1
        
        return pd.DataFrame(predictions)
    
    def detect_market_regime(self, test_df):
        """
        Detect market regime: Bull, Bear, or Sideways
        Dựa trên: Price change, Trend, RSI
        """
        price_change = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0] * 100
        avg_trend = test_df['trend'].mean()
        avg_rsi = test_df['rsi'].mean()
        
        print(f"\nMarket Detection:")
        print(f"  Price Change: {price_change:.2f}%")
        print(f"  Avg Trend: {avg_trend:.3f}")
        print(f"  Avg RSI: {avg_rsi:.2f}")
        
        # Detect regime (RELAXED thresholds)
        if price_change > 1 and avg_trend > 0.2:
            regime = 'BULL'
        elif price_change < -2 and avg_trend < -0.3:
            regime = 'BEAR'
        else:
            regime = 'SIDEWAYS'
        
        print(f"  Detected Regime: {regime}")
        
        return regime, {
            'price_change': price_change,
            'avg_trend': avg_trend,
            'avg_rsi': avg_rsi
        }
    
    def apply_market_adaptive_rules(self, pred_df, test_df, market_regime):
        """
        Apply different rules based on market regime
        
        BULL: Relax Buy constraints, encourage profit-taking
        BEAR: Strict Buy blocking (existing rules)
        SIDEWAYS: Moderate, range trading
        """
        print(f"\n" + "="*60)
        print(f"APPLYING MARKET-ADAPTIVE RULES ({market_regime})")
        print("="*60)
        
        improved_actions = []
        corrections = []
        
        for idx, row in pred_df.iterrows():
            raw_action = row['raw_action']
            improved_action = raw_action
            reason = ""
            
            # Get indicators
            trend = test_df.iloc[idx]['trend']
            rsi = test_df.iloc[idx]['rsi']
            
            if idx > 0:
                prev_price = test_df.iloc[idx - 1]['close']
                curr_price = test_df.iloc[idx]['close']
                price_change_pct = (curr_price - prev_price) / prev_price
            else:
                price_change_pct = 0
            
            # ========== BULL MARKET RULES ==========
            if market_regime == 'BULL':
                # Rule B0: NEVER sell immediately after buy (prevent panic sells)
                if raw_action == 2 and row['position'] == 1 and idx <= 2:
                    improved_action = 0
                    reason = f"Bull-R0: Block immediate sell after Buy (hour={idx})"
                    corrections.append({
                        'hour': idx,
                        'raw': 'Sell',
                        'improved': 'Hold',
                        'reason': reason
                    })
                
                # Rule B1: Take profit when overbought OR end of day
                elif raw_action == 2 and row['position'] == 1:
                    # Allow Sell if: (1) RSI very high, (2) trend weakening, (3) last hour
                    if rsi > 75 or trend < 0 or idx >= 20:
                        # Keep the Sell
                        pass
                    # Block Sell if still in strong uptrend with moderate RSI
                    elif trend > 0.3 and rsi < 75 and idx < 20:
                        improved_action = 0  # Hold instead
                        reason = f"Bull-R1: Hold instead of early Sell (trend={trend:.2f}, RSI={rsi:.1f})"
                        corrections.append({
                            'hour': idx,
                            'raw': 'Sell',
                            'improved': 'Hold',
                            'reason': reason
                        })
                
                # Rule B2: STRONG - Encourage Buy when DQN is Hold but conditions favor buying
                elif raw_action == 0 and row['position'] == 0:
                    # Relaxed Buy conditions for Bull market
                    # B2a: Buy if moderate uptrend (even if RSI is higher)
                    if trend > 0.5 and rsi < 80:
                        improved_action = 1
                        reason = f"Bull-R2a: Buy in strong uptrend (trend={trend:.2f}, RSI={rsi:.1f})"
                        corrections.append({
                            'hour': idx,
                            'raw': 'Hold',
                            'improved': 'Buy',
                            'reason': reason
                        })
                    # B2b: Buy if any uptrend with reasonable RSI
                    elif trend > 0.2 and rsi < 70:
                        improved_action = 1
                        reason = f"Bull-R2b: Buy on uptrend (trend={trend:.2f}, RSI={rsi:.1f})"
                        corrections.append({
                            'hour': idx,
                            'raw': 'Hold',
                            'improved': 'Buy',
                            'reason': reason
                        })
                    # B2c: Buy on dips (moderate trend, low RSI)
                    elif trend > 0.1 and rsi < 45:
                        improved_action = 1
                        reason = f"Bull-R2c: Buy on dip (trend={trend:.2f}, RSI={rsi:.1f})"
                        corrections.append({
                            'hour': idx,
                            'raw': 'Hold',
                            'improved': 'Buy',
                            'reason': reason
                        })
                    # B2d: AGGRESSIVE - Buy early if starting with downtrend but will reverse
                    elif idx == 0 and rsi < 75:  # First hour, be aggressive
                        improved_action = 1
                        reason = f"Bull-R2d: Early entry in Bull day (trend={trend:.2f}, RSI={rsi:.1f})"
                        corrections.append({
                            'hour': idx,
                            'raw': 'Hold',
                            'improved': 'Buy',
                            'reason': reason
                        })
                
                # Rule B3: Take profit when overbought
                elif rsi > 75 and trend < 0.1 and row['position'] == 1:
                    if raw_action != 2:
                        improved_action = 2  # Sell to take profit
                        reason = f"Bull-R3: Take profit overbought (RSI={rsi:.1f}, trend={trend:.2f})"
                        corrections.append({
                            'hour': idx,
                            'raw': self.action_names[raw_action],
                            'improved': 'Sell',
                            'reason': reason
                        })
            
            # ========== BEAR MARKET RULES ==========
            elif market_regime == 'BEAR':
                # Rule BR1: STRICT - Block Buy in strong downtrend
                if raw_action == 1 and trend < -0.5:
                    improved_action = 0
                    reason = f"Bear-R1: Block Buy in strong downtrend (trend={trend:.2f})"
                    corrections.append({
                        'hour': idx,
                        'raw': 'Buy',
                        'improved': 'Hold',
                        'reason': reason
                    })
                
                # Rule BR2: Block falling knife
                elif raw_action == 1 and rsi < 30 and trend < 0:
                    improved_action = 0
                    reason = f"Bear-R2: Avoid falling knife (RSI={rsi:.1f}, trend={trend:.2f})"
                    corrections.append({
                        'hour': idx,
                        'raw': 'Buy',
                        'improved': 'Hold',
                        'reason': reason
                    })
                
                # Rule BR3: Encourage Sell/Exit
                elif rsi > 60 and trend < -0.3 and row['position'] == 1:
                    if raw_action != 2:
                        improved_action = 2
                        reason = f"Bear-R3: Exit position (RSI={rsi:.1f}, trend={trend:.2f})"
                        corrections.append({
                            'hour': idx,
                            'raw': self.action_names[raw_action],
                            'improved': 'Sell',
                            'reason': reason
                        })
            
            # ========== SIDEWAYS MARKET RULES ==========
            else:  # SIDEWAYS
                # Rule S1: Buy at support (RSI oversold)
                if raw_action == 0 and rsi < 35 and row['position'] == 0:
                    improved_action = 1
                    reason = f"Sideways-R1: Buy at support (RSI={rsi:.1f})"
                    corrections.append({
                        'hour': idx,
                        'raw': 'Hold',
                        'improved': 'Buy',
                        'reason': reason
                    })
                
                # Rule S2: Sell at resistance (RSI overbought)
                elif rsi > 65 and row['position'] == 1:
                    if raw_action != 2:
                        improved_action = 2
                        reason = f"Sideways-R2: Sell at resistance (RSI={rsi:.1f})"
                        corrections.append({
                            'hour': idx,
                            'raw': self.action_names[raw_action],
                            'improved': 'Sell',
                            'reason': reason
                        })
                
                # Rule S3: Hold in neutral zone
                elif 40 < rsi < 60 and abs(trend) < 0.2:
                    if raw_action == 1:
                        improved_action = 0
                        reason = f"Sideways-R3: Hold in neutral (RSI={rsi:.1f}, trend={trend:.2f})"
                        corrections.append({
                            'hour': idx,
                            'raw': 'Buy',
                            'improved': 'Hold',
                            'reason': reason
                        })
            
            improved_actions.append(improved_action)
        
        pred_df['improved_action'] = improved_actions
        pred_df['improved_action_name'] = pred_df['improved_action'].map(self.action_names)
        
        # ⭐ FORCE EXIT at end of day if still holding
        last_idx = len(pred_df) - 1
        if pred_df.iloc[last_idx]['position'] == 1 and pred_df.iloc[last_idx]['improved_action'] != 2:
            pred_df.at[last_idx, 'improved_action'] = 2
            pred_df.at[last_idx, 'improved_action_name'] = 'Sell'
            corrections.append({
                'hour': last_idx,
                'raw': self.action_names[pred_df.iloc[last_idx]['raw_action']],
                'improved': 'Sell',
                'reason': 'FORCE EXIT: End of day closing position'
            })
        
        print(f"\nTotal corrections: {len(corrections)}")
        if len(corrections) > 0:
            print("\nCorrections made:")
            for c in corrections[:10]:  # Show first 10
                print(f"  Hour {c['hour']:2d}: {c['raw']:4s} -> {c['improved']:4s} | {c['reason']}")
            if len(corrections) > 10:
                print(f"  ... and {len(corrections) - 10} more")
        else:
            print("  No corrections needed!")
        
        return pred_df, corrections
    
    def execute_improved_strategy(self, pred_df, test_df):
        """Execute strategy với improved actions"""
        balance = 10000.0
        holdings = 0.0
        position = 0
        transaction_fee = 0.001  # 0.1% fee
        
        portfolio_history = []
        
        for idx, row in pred_df.iterrows():
            action = row['improved_action']
            price = row['price']
            
            if action == 1 and position == 0:  # Buy
                # Buy all with balance, after deducting fee
                holdings = balance / (price * (1 + transaction_fee))
                balance = 0
                position = 1
            
            elif action == 2 and position == 1:  # Sell
                # Sell all holdings, after deducting fee
                balance = holdings * price * (1 - transaction_fee)
                holdings = 0
                position = 0
            
            portfolio_value = balance + holdings * price
            portfolio_history.append(portfolio_value)
        
        pred_df['improved_portfolio'] = portfolio_history
        pred_df['improved_balance'] = balance
        pred_df['improved_holdings'] = holdings
        
        return pred_df
    
    def calculate_metrics(self, pred_df, test_df):
        """Tính metrics"""
        price_changes = test_df['close'].pct_change().fillna(0)
        ideal_actions = []
        for change in price_changes:
            if change > 0.01:
                ideal_actions.append(1)
            elif change < -0.01:
                ideal_actions.append(2)
            else:
                ideal_actions.append(0)
        
        ideal_actions = np.array(ideal_actions[:len(pred_df)])
        improved_actions = pred_df['improved_action'].values
        
        mae = mean_absolute_error(ideal_actions, improved_actions)
        mse = mean_squared_error(ideal_actions, improved_actions)
        rmse = np.sqrt(mse)
        
        mape_vals = []
        for true, pred in zip(ideal_actions, improved_actions):
            if true != 0:
                mape_vals.append(abs((true - pred) / true))
        mape = np.mean(mape_vals) * 100 if mape_vals else 0
        
        r2 = r2_score(ideal_actions, improved_actions)
        accuracy = (improved_actions == ideal_actions).sum() / len(improved_actions) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Accuracy': accuracy
        }
    
    def compare_results(self, pred_df, test_df):
        """So sánh Raw vs Improved"""
        print("\n" + "="*60)
        print("COMPARISON: RAW vs ADAPTIVE")
        print("="*60)
        
        raw_dist = pred_df['raw_action_name'].value_counts()
        improved_dist = pred_df['improved_action_name'].value_counts()
        
        print("\nAction Distribution:")
        print(f"{'Action':<10} {'Raw':<15} {'Adaptive':<15}")
        print("-"*40)
        for action in ['Hold', 'Buy', 'Sell']:
            raw_count = raw_dist.get(action, 0)
            imp_count = improved_dist.get(action, 0)
            print(f"{action:<10} {raw_count:2d} ({raw_count/24*100:4.1f}%)    "
                  f"{imp_count:2d} ({imp_count/24*100:4.1f}%)")
        
        raw_final = pred_df['balance'].iloc[-1] + pred_df['holdings'].iloc[-1] * pred_df['price'].iloc[-1]
        improved_final = pred_df['improved_portfolio'].iloc[-1]
        
        raw_return = (raw_final - 10000) / 10000 * 100
        improved_return = (improved_final - 10000) / 10000 * 100
        
        print("\nPerformance:")
        print(f"{'Metric':<20} {'Raw':<15} {'Adaptive':<15}")
        print("-"*50)
        print(f"{'Final Value':<20} ${raw_final:>12,.2f}  ${improved_final:>12,.2f}")
        print(f"{'Return %':<20} {raw_return:>12.2f}%  {improved_return:>12.2f}%")
        print(f"{'Improvement':<20} {'':<15} {improved_return - raw_return:>+12.2f}%")
        
        return {
            'raw_return': raw_return,
            'improved_return': improved_return,
            'improvement': improved_return - raw_return
        }
    
    def analyze_market(self, test_df):
        """Phân tích môi trường thị trường"""
        price_change = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0] * 100
        avg_trend = test_df['trend'].mean()
        avg_rsi = test_df['rsi'].mean()
        avg_volatility = test_df['volatility'].mean()
        
        if price_change > 2 and avg_trend > 0:
            market_type = 'Strong Bull Market (Tăng mạnh)'
        elif price_change > 0.5 and avg_trend > 0:
            market_type = 'Bull Market (Tăng)'
        elif price_change < -2 and avg_trend < 0:
            market_type = 'Strong Bear Market (Giảm mạnh)'
        elif price_change < -0.5 and avg_trend < 0:
            market_type = 'Bear Market (Giảm)'
        else:
            market_type = 'Sideways Market (Đi ngang)'
        
        return {
            'market_type': market_type,
            'price_change': price_change,
            'avg_trend': avg_trend,
            'avg_rsi': avg_rsi,
            'avg_volatility': avg_volatility
        }
    
    def run(self, test_date=None):
        """Run adaptive prediction pipeline"""
        print("="*60)
        print("MARKET-ADAPTIVE PREDICTION")
        print("="*60)
        
        test_df = self.load_data(test_date)
        print(f"\nTest date: {test_df['timestamp'].iloc[0].strftime('%Y-%m-%d')}")
        
        print("\nLoading DQN model...")
        agent, env = self.load_model(test_df)
        
        print("\nGetting raw predictions from DQN...")
        pred_df = self.get_raw_predictions(agent, test_df)
        
        # Detect market regime
        market_regime, regime_stats = self.detect_market_regime(test_df)
        
        # Apply market-adaptive rules
        pred_df, corrections = self.apply_market_adaptive_rules(pred_df, test_df, market_regime)
        
        # Execute strategy
        print("\nExecuting adaptive strategy...")
        pred_df = self.execute_improved_strategy(pred_df, test_df)
        
        # Compare results
        comparison = self.compare_results(pred_df, test_df)
        
        # Calculate metrics
        metrics = self.calculate_metrics(pred_df, test_df)
        
        # Analyze market
        market_info = self.analyze_market(test_df)
        
        print("\n" + "="*60)
        print("COMPLETED!")
        print("="*60)
        print(f"Market Regime: {market_regime}")
        print(f"Improvement: {comparison['improvement']:+.2f}%")
        print(f"MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R²: {metrics['R2']:.4f}")
        print(f"Accuracy: {metrics['Accuracy']:.2f}%")
        
        return pred_df, comparison, metrics, market_regime

if __name__ == '__main__':
    CHECKPOINT_PATH = 'src/checkpoints_dqn/checkpoint_best.pkl'
    DATA_PATH = 'data/raw/multi_coin_1h.csv'
    
    # Test on all 3 regimes
    test_dates = {
        'Bull': '2020-04-06',
        'Bear': '2020-03-12',
        'Sideways': '2024-09-30'
    }
    
    predictor = MarketAdaptivePredictor(CHECKPOINT_PATH, DATA_PATH)
    
    for market_name, test_date in test_dates.items():
        print("\n" + "="*80)
        print(f"TESTING {market_name.upper()} MARKET - {test_date}")
        print("="*80)
        
        results = predictor.run(test_date=test_date)
