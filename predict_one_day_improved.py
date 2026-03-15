"""
IMPROVED PREDICTION - Không cần train lại
Sử dụng model hiện tại + Post-processing rules
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

class ImprovedPredictor:
    """Cải thiện predictions bằng post-processing rules"""
    
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
            
            # Raw prediction từ DQN
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
    
    def apply_smart_rules(self, pred_df, test_df):
        """
        Apply MARKET-ADAPTIVE post-processing rules
        Rules thay đổi dựa trên market regime (Bull/Bear/Sideways)
        """
        # Detect market regime first
        market_regime, regime_stats = self.detect_market_regime(test_df)
        
        print("\n" + "="*60)
        print(f"APPLYING MARKET-ADAPTIVE RULES ({market_regime})")
        print("="*60)
        
        improved_actions = []
        corrections = []
        
        for idx, row in pred_df.iterrows():
            raw_action = row['raw_action']
            improved_action = raw_action  # Default: giữ nguyên
            reason = ""
            
            # Get market indicators
            trend = test_df.iloc[idx]['trend']
            rsi = test_df.iloc[idx]['rsi']
            
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
                    # Allow Sell if: (1) RSI extremely high OR (2) trend reversing OR (3) near end of day
                    if (rsi > 85) or (rsi > 80 and trend < 0.5) or (trend < 0) or (idx >= 22):
                        # Keep the Sell
                        pass
                    # Block Sell if still in uptrend with reasonable RSI
                    elif trend > 0.3 and rsi <= 85 and idx < 22:
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
        
        # Add improved actions to dataframe
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
        
        # Print corrections
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
        position = 0  # 0 = cash, 1 = holding BTC
        transaction_fee = 0.001
        
        portfolio_history = []
        
        for idx, row in pred_df.iterrows():
            action = row['improved_action']
            price = row['price']
            
            # Execute action
            if action == 1 and position == 0:  # Buy
                # Buy all with balance, after deducting fee
                holdings = balance / (price * (1 + transaction_fee))
                balance = 0
                position = 1
            
            elif action == 2 and position == 1:  # Sell
                balance = holdings * price * (1 - transaction_fee)
                holdings = 0
                position = 0
            
            # Calculate portfolio value
            portfolio_value = balance + holdings * price
            portfolio_history.append(portfolio_value)
        
        pred_df['improved_portfolio'] = portfolio_history
        pred_df['improved_balance'] = balance
        pred_df['improved_holdings'] = holdings
        
        return pred_df
    
    def calculate_metrics(self, pred_df, test_df):
        """
        Tính metrics MAE, MSE, RMSE, MAPE, R2 cho improved predictions
        So sánh improved actions với ideal actions
        """
        # Tính ideal actions dựa trên price direction
        price_changes = test_df['close'].pct_change().fillna(0)
        ideal_actions = []
        for change in price_changes:
            if change > 0.01:  # Price up > 1%
                ideal_actions.append(1)  # Should Buy
            elif change < -0.01:  # Price down > 1%
                ideal_actions.append(2)  # Should Sell
            else:
                ideal_actions.append(0)  # Hold
        
        ideal_actions = np.array(ideal_actions[:len(pred_df)])
        improved_actions = pred_df['improved_action'].values
        
        # Calculate metrics
        mae = mean_absolute_error(ideal_actions, improved_actions)
        mse = mean_squared_error(ideal_actions, improved_actions)
        rmse = np.sqrt(mse)
        
        # MAPE
        mape_vals = []
        for true, pred in zip(ideal_actions, improved_actions):
            if true != 0:
                mape_vals.append(abs((true - pred) / true))
        mape = np.mean(mape_vals) * 100 if mape_vals else 0
        
        # R2 score
        r2 = r2_score(ideal_actions, improved_actions)
        
        # Accuracy
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
        print("COMPARISON: RAW vs IMPROVED")
        print("="*60)
        
        # Action distribution
        raw_dist = pred_df['raw_action_name'].value_counts()
        improved_dist = pred_df['improved_action_name'].value_counts()
        
        print("\nAction Distribution:")
        print(f"{'Action':<10} {'Raw':<15} {'Improved':<15}")
        print("-"*40)
        for action in ['Hold', 'Buy', 'Sell']:
            raw_count = raw_dist.get(action, 0)
            imp_count = improved_dist.get(action, 0)
            print(f"{action:<10} {raw_count:2d} ({raw_count/24*100:4.1f}%)    "
                  f"{imp_count:2d} ({imp_count/24*100:4.1f}%)")
        
        # Performance
        # Tính portfolio cho raw (giả sử từ predictions gốc)
        raw_final = pred_df['balance'].iloc[-1] + pred_df['holdings'].iloc[-1] * pred_df['price'].iloc[-1]
        improved_final = pred_df['improved_portfolio'].iloc[-1]
        
        raw_return = (raw_final - 10000) / 10000 * 100
        improved_return = (improved_final - 10000) / 10000 * 100
        
        print("\nPerformance:")
        print(f"{'Metric':<20} {'Raw':<15} {'Improved':<15}")
        print("-"*50)
        print(f"{'Final Value':<20} ${raw_final:>12,.2f}  ${improved_final:>12,.2f}")
        print(f"{'Return %':<20} {raw_return:>12.2f}%  {improved_return:>12.2f}%")
        print(f"{'Improvement':<20} {'':<15} {improved_return - raw_return:>+12.2f}%")
        
        return {
            'raw_return': raw_return,
            'improved_return': improved_return,
            'improvement': improved_return - raw_return
        }
    
    def plot_comparison(self, pred_df, test_df, save_path='results/charts/'):
        """Vẽ so sánh Raw vs Improved"""
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Panel 1: Price + Actions
        ax1 = axes[0, 0]
        ax1.plot(pred_df['timestamp'], pred_df['price'], 'b-', linewidth=2, label='BTC Price')
        
        # Raw actions
        for action, marker, color in [(1, '^', 'lightgreen'), (2, 'v', 'lightcoral')]:
            mask = pred_df['raw_action'] == action
            ax1.scatter(pred_df[mask]['timestamp'], pred_df[mask]['price'],
                       c=color, marker=marker, s=100, alpha=0.5, 
                       label=f'Raw {self.action_names[action]}')
        
        # Improved actions
        for action, marker, color in [(1, '^', 'darkgreen'), (2, 'v', 'darkred')]:
            mask = pred_df['improved_action'] == action
            ax1.scatter(pred_df[mask]['timestamp'], pred_df[mask]['price'],
                       c=color, marker=marker, s=150, alpha=0.9, edgecolors='black',
                       label=f'Improved {self.action_names[action]}', linewidths=2)
        
        ax1.set_title('Price + Actions (Raw vs Improved)', fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Panel 2: Action Distribution
        ax2 = axes[0, 1]
        x = np.arange(3)
        width = 0.35
        
        raw_counts = [pred_df['raw_action'].value_counts().get(i, 0) for i in range(3)]
        imp_counts = [pred_df['improved_action'].value_counts().get(i, 0) for i in range(3)]
        
        ax2.bar(x - width/2, raw_counts, width, label='Raw', alpha=0.7)
        ax2.bar(x + width/2, imp_counts, width, label='Improved', alpha=0.7)
        ax2.set_xlabel('Action')
        ax2.set_ylabel('Count')
        ax2.set_title('Action Distribution Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Hold', 'Buy', 'Sell'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Portfolio Value
        ax3 = axes[1, 0]
        # Raw portfolio (approximation)
        ax3.plot(pred_df['timestamp'], 
                pred_df['balance'] + pred_df['holdings'] * pred_df['price'],
                'r--', linewidth=2, label='Raw Strategy', alpha=0.7)
        ax3.plot(pred_df['timestamp'], pred_df['improved_portfolio'],
                'g-', linewidth=2, label='Improved Strategy')
        ax3.axhline(y=10000, color='black', linestyle=':', label='Initial $10,000')
        ax3.set_title('Portfolio Value Comparison', fontweight='bold')
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Panel 4: Market Indicators
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        
        ax4.plot(test_df['timestamp'][:24], test_df['rsi'][:24], 
                'b-', label='RSI', linewidth=2)
        ax4.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        ax4.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        ax4.set_ylabel('RSI', color='b')
        ax4.set_ylim([0, 100])
        
        ax4_twin.plot(test_df['timestamp'][:24], test_df['trend'][:24],
                     'r-', label='Trend', linewidth=2)
        ax4_twin.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4_twin.set_ylabel('Trend', color='r')
        
        ax4.set_title('Market Indicators', fontweight='bold')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_file = os.path.join(save_path, 'improved_prediction_comparison.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison chart: {save_file}")
        plt.close()
    
    def plot_detailed_report(self, pred_df, test_df, comparison, market_info, metrics, save_path='results/charts/'):
        """Vẽ báo cáo chi tiết 6-panel giống one_day_prediction_dqn.png"""
        os.makedirs(save_path, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Price + Improved Actions
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(pred_df['timestamp'], pred_df['price'], 'b-', linewidth=2, label='BTC Price')
        
        buy_mask = pred_df['improved_action'] == 1
        sell_mask = pred_df['improved_action'] == 2
        
        ax1.scatter(pred_df[buy_mask]['timestamp'], pred_df[buy_mask]['price'], 
                   c='green', marker='^', s=200, label='Buy', zorder=5)
        ax1.scatter(pred_df[sell_mask]['timestamp'], pred_df[sell_mask]['price'], 
                   c='red', marker='v', s=200, label='Sell', zorder=5)
        
        ax1.set_title('BTC Price với Hành Động IMPROVED', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Thời gian')
        ax1.set_ylabel('Giá (USD)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Panel 2: Portfolio Value
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(pred_df['timestamp'], pred_df['improved_portfolio'], 'g-', linewidth=2)
        ax2.axhline(y=10000, color='r', linestyle='--', label='Initial Balance ($10,000)')
        ax2.fill_between(pred_df['timestamp'], 10000, pred_df['improved_portfolio'], 
                         where=pred_df['improved_portfolio']>=10000, alpha=0.3, color='green', label='Profit')
        ax2.fill_between(pred_df['timestamp'], 10000, pred_df['improved_portfolio'], 
                         where=pred_df['improved_portfolio']<10000, alpha=0.3, color='red', label='Loss')
        
        ax2.set_title('Giá Trị Danh Mục Theo Thời Gian (IMPROVED)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Thời gian')
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Panel 3: Rewards (từ raw predictions)
        ax3 = fig.add_subplot(gs[2, :2])
        colors = ['blue' if r >= 0 else 'red' for r in pred_df['reward']]
        ax3.bar(pred_df['timestamp'], pred_df['reward'], color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Rewards Theo Thời Gian', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Thời gian')
        ax3.set_ylabel('Reward')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
        
        # Panel 4: Technical Indicators
        ax4 = fig.add_subplot(gs[3, :2])
        ax4_2 = ax4.twinx()
        
        ax4.plot(test_df['timestamp'][:len(pred_df)], test_df['rsi'][:len(pred_df)], 
                'b-', label='RSI', linewidth=2)
        ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax4.set_ylabel('RSI', color='b')
        ax4.tick_params(axis='y', labelcolor='b')
        ax4.set_ylim([0, 100])
        
        ax4_2.plot(test_df['timestamp'][:len(pred_df)], test_df['trend'][:len(pred_df)], 
                  'r-', label='Trend', linewidth=2)
        ax4_2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4_2.set_ylabel('Trend', color='r')
        ax4_2.tick_params(axis='y', labelcolor='r')
        
        ax4.set_title('Chỉ Báo Kỹ Thuật', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Thời gian')
        ax4.legend(loc='upper left')
        ax4_2.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # Panel 5: Action Distribution (Improved)
        ax5 = fig.add_subplot(gs[0, 2])
        action_counts = pred_df['improved_action_name'].value_counts()
        colors_pie = {'Hold': 'blue', 'Buy': 'green', 'Sell': 'red'}
        colors_list = [colors_pie.get(action, 'gray') for action in action_counts.index]
        
        ax5.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%',
               colors=colors_list, startangle=90)
        ax5.set_title('Phân Bố Hành Động\n(IMPROVED)', fontsize=14, fontweight='bold')
        
        # Panel 6: Metrics Table
        ax6 = fig.add_subplot(gs[1:, 2])
        ax6.axis('off')
        
        final_value = pred_df['improved_portfolio'].iloc[-1]
        total_return = (final_value - 10000) / 10000 * 100
        num_trades = (pred_df['improved_action'] != 0).sum()
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Strategy', 'IMPROVED'],
            ['', ''],
            ['MAE', f"{metrics['MAE']:.4f}"],
            ['MSE', f"{metrics['MSE']:.4f}"],
            ['RMSE', f"{metrics['RMSE']:.4f}"],
            ['MAPE', f"{metrics['MAPE']:.2f}%"],
            ['R² Score', f"{metrics['R2']:.4f}"],
            ['Accuracy', f"{metrics['Accuracy']:.2f}%"],
            ['', ''],
            ['Total Return', f"{total_return:.2f}%"],
            ['Final Balance', f"${final_value:.2f}"],
            ['Profit/Loss', f"${final_value - 10000:.2f}"],
            ['Num Trades', f"{num_trades}"],
            ['', ''],
            ['Raw Return', f"{comparison['raw_return']:.2f}%"],
            ['Improvement', f"{comparison['improvement']:+.2f}%"],
            ['', ''],
            ['Market Type', market_info['market_type'].split('(')[0]],
            ['Price Change', f"{market_info['price_change']:.2f}%"],
            ['Avg Trend', f"{market_info['avg_trend']:.3f}"],
            ['Avg RSI', f"{market_info['avg_rsi']:.2f}"],
        ]
        
        table = ax6.table(cellText=metrics_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style metrics rows
        for i in range(1, len(metrics_data)):
            if metrics_data[i][0] == '':
                continue
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#ffffff')
        
        ax6.set_title('Metrics & Market Info', fontsize=14, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle(f'DỰ BÁO HÀNH ĐỘNG GIAO DỊCH BTC - IMPROVED STRATEGY\n{pred_df["timestamp"].iloc[0].strftime("%Y-%m-%d")}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        save_file = os.path.join(save_path, 'one_day_prediction_improved.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved detailed report chart: {save_file}")
        
        plt.close()
    
    def analyze_market(self, test_df):
        """Phân tích môi trường thị trường"""
        price_change = (test_df['close'].iloc[-1] - test_df['close'].iloc[0]) / test_df['close'].iloc[0] * 100
        avg_trend = test_df['trend'].mean()
        avg_rsi = test_df['rsi'].mean()
        avg_volatility = test_df['volatility'].mean()
        
        # Determine market type
        if price_change > 2 and avg_trend > 0:
            market_type = 'Strong Bull Market (Tăng mạnh)'
            optimal_strategy = 'Buy early, Hold, Sell at peak'
        elif price_change > 0.5 and avg_trend > 0:
            market_type = 'Bull Market (Tăng)'
            optimal_strategy = 'Buy on dips, Hold'
        elif price_change < -2 and avg_trend < 0:
            market_type = 'Strong Bear Market (Giảm mạnh)'
            optimal_strategy = 'Sell early, Hold cash, Wait for reversal'
        elif price_change < -0.5 and avg_trend < 0:
            market_type = 'Bear Market (Giảm)'
            optimal_strategy = 'Reduce positions, Hold cash'
        else:
            market_type = 'Sideways Market (Đi ngang)'
            optimal_strategy = 'Range trading, Quick profits'
        
        return {
            'market_type': market_type,
            'price_change': price_change,
            'avg_trend': avg_trend,
            'avg_rsi': avg_rsi,
            'avg_volatility': avg_volatility,
            'optimal_strategy': optimal_strategy
        }
    
    def run(self, test_date=None):
        """Run improved prediction pipeline"""
        print("="*60)
        print("IMPROVED PREDICTION - NO RETRAINING NEEDED")
        print("="*60)
        
        # Load data
        test_df = self.load_data(test_date)
        print(f"\nTest date: {test_df['timestamp'].iloc[0].strftime('%Y-%m-%d')}")
        
        # Load model
        print("\nLoading DQN model...")
        agent, env = self.load_model(test_df)
        
        # Get raw predictions
        print("\nGetting raw predictions from DQN...")
        pred_df = self.get_raw_predictions(agent, test_df)
        
        # Apply smart rules
        pred_df, corrections = self.apply_smart_rules(pred_df, test_df)
        
        # Execute improved strategy
        print("\nExecuting improved strategy...")
        pred_df = self.execute_improved_strategy(pred_df, test_df)
        
        # Compare results
        comparison = self.compare_results(pred_df, test_df)
        
        # Calculate metrics (MAE, MSE, RMSE, MAPE, R2)
        metrics = self.calculate_metrics(pred_df, test_df)
        
        # Analyze market
        market_info = self.analyze_market(test_df)
        
        # Plot comparison (4-panel)
        self.plot_comparison(pred_df, test_df)
        
        # Plot detailed report (6-panel like original)
        self.plot_detailed_report(pred_df, test_df, comparison, market_info, metrics)
        
        print("\n" + "="*60)
        print("COMPLETED!")
        print("="*60)
        print(f"Improvement: {comparison['improvement']:+.2f}%")
        print(f"MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R²: {metrics['R2']:.4f}")
        print(f"Accuracy: {metrics['Accuracy']:.2f}%")
        print(f"Chart saved: results/charts/improved_prediction_comparison.png")
        print(f"Detailed chart: results/charts/one_day_prediction_improved.png")
        
        return pred_df, comparison

if __name__ == '__main__':
    CHECKPOINT_PATH = 'src/checkpoints_dqn/checkpoint_best.pkl'
    DATA_PATH = 'data/raw/multi_coin_1h.csv'
    
    # Test dates
    # Bear: '2020-03-12', '2019-06-27'
    # Bull: '2020-04-06', '2020-04-22'
    TEST_DATE = '2020-04-06'  # Bull Market +6.76%
    
    predictor = ImprovedPredictor(CHECKPOINT_PATH, DATA_PATH)
    results, comparison = predictor.run(test_date=TEST_DATE)
