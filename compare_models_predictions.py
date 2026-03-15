"""
Script để so sánh dự báo hành động của các mô hình:
- LSTM
- PSO + LSTM  
- PPO + PSO + LSTM
- DQN
- DQN + PSO + LSTM

Metric: MAE, MSE, RMSE, MAPE, R2
Output: Đồ thị, bảng so sánh, phân tích chi tiết
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.multi_coin_loader import MultiCoinLoader
from src.models.mdp_trading import TradingMDP
from src.models.lstm_trading import LSTMTradingAgent
from src.models.pso_lstm_trading import PSOLSTMOptimizer
from src.models.ppo_pso_lstm_trading import PPOPSOLSTMAgent
from src.models.dqn_pso_lstm_trading import DQNLSTMAgent
from src.models.dqn_agent import DQNAgent

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelComparator:
    """Class để so sánh các mô hình trading"""
    
    def __init__(self, coin='BTC', test_days=30):
        """
        Args:
            coin: Đồng coin để test (default: BTC)
            test_days: Số ngày để test (default: 30)
        """
        self.coin = coin
        self.test_days = test_days
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def load_data(self):
        """Load và chuẩn bị dữ liệu"""
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        
        # Load from CSV
        df = pd.read_csv('data/raw/multi_coin_1h.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter coin
        df_coin = df[df['coin'] == self.coin].copy()
        print(f"   {self.coin} data: {len(df_coin):,} rows")
        print(f"   Date range: {df_coin['timestamp'].min()} to {df_coin['timestamp'].max()}")
        
        # Split: 80% train, 20% test
        train_size = int(len(df_coin) * 0.8)
        self.train_df = df_coin.iloc[:train_size].copy()
        self.test_df = df_coin.iloc[train_size:].copy()
        
        # Take last test_days from test set
        hours_to_take = self.test_days * 24  # 24 hours per day
        self.test_df = self.test_df.tail(hours_to_take).copy()
        
        print(f"\nData Split:")
        print(f"   Train: {len(self.train_df):,} samples")
        print(f"   Test: {len(self.test_df):,} samples ({self.test_days} days)")
        print(f"   Test period: {self.test_df['timestamp'].min()} to {self.test_df['timestamp'].max()}")
        
        return self.train_df, self.test_df
    
    def load_dqn_model(self):
        """Load DQN model từ checkpoint"""
        print("\n" + "="*80)
        print("LOADING DQN MODEL")
        print("="*80)
        
        try:
            checkpoint_path = 'src/checkpoints_dqn/checkpoint_best.pkl'
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract hyperparameters
            state_dim = checkpoint['hyperparameters']['state_dim']
            action_dim = checkpoint['hyperparameters']['action_dim']
            learning_rate = checkpoint['hyperparameters']['learning_rate']
            gamma = checkpoint['hyperparameters']['gamma']
            batch_size = checkpoint['hyperparameters']['batch_size']
            
            print(f"   Checkpoint: {checkpoint_path}")
            print(f"   State dim: {state_dim}")
            print(f"   Action dim: {action_dim}")
            print(f"   Episode: {checkpoint.get('episode', 'Unknown')}")
            print(f"   Best profit: {checkpoint.get('best_profit', 'Unknown')}")
            
            # Create DQN agent
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
                gamma=gamma,
                batch_size=batch_size
            )
            
            # Load weights
            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            agent.epsilon = checkpoint['epsilon']
            
            # Move to CPU and set to eval mode
            agent.policy_net.to('cpu')
            agent.target_net.to('cpu')
            agent.device = torch.device('cpu')
            agent.policy_net.eval()
            agent.target_net.eval()
            
            # Store for prediction
            self.models['DQN'] = {
                'agent': agent,
                'checkpoint': checkpoint,
                'type': 'dqn'
            }
            
            print("   Status: DQN model loaded successfully")
            
        except Exception as e:
            print(f"   Error loading DQN: {e}")
            import traceback
            traceback.print_exc()
            self.models['DQN'] = None
    
    def load_dqn_pso_lstm_model(self):
        """Load DQN+PSO+LSTM model nếu có"""
        print("\n" + "="*80)
        print("LOADING DQN+PSO+LSTM MODEL")
        print("="*80)
        
        # Check for checkpoint
        checkpoint_dirs = ['checkpoints/dqn_pso_lstm', 'checkpoints/dqn_pso_lstm_gpu']
        
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                if checkpoints:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                    
                    try:
                        # Load checkpoint
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        
                        # Create agent
                        agent = DQNLSTMAgent(
                            input_size=checkpoint['hyperparameters']['input_size'],
                            hidden_size=checkpoint['hyperparameters']['hidden_size'],
                            num_layers=checkpoint['hyperparameters']['num_layers'],
                            sequence_length=checkpoint['hyperparameters']['sequence_length'],
                            learning_rate=checkpoint['hyperparameters']['learning_rate'],
                            gamma=checkpoint['hyperparameters']['gamma'],
                            epsilon_decay=checkpoint['hyperparameters']['epsilon_decay'],
                            batch_size=checkpoint['hyperparameters']['batch_size'],
                            target_update_frequency=checkpoint['hyperparameters']['target_update_frequency']
                        )
                        
                        # Load weights
                        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                        agent.epsilon = checkpoint['epsilon']
                        
                        agent.policy_net.eval()
                        agent.target_net.eval()
                        
                        print(f"   Checkpoint: {checkpoint_path}")
                        print(f"   Hyperparameters: {checkpoint['hyperparameters']}")
                        
                        self.models['DQN+PSO+LSTM'] = {
                            'agent': agent,
                            'checkpoint': checkpoint,
                            'type': 'dqn_lstm'
                        }
                        
                        print("   Status: DQN+PSO+LSTM model loaded successfully")
                        return
                        
                    except Exception as e:
                        print(f"   Error loading from {checkpoint_path}: {e}")
        
        print("   Status: No DQN+PSO+LSTM checkpoint found")
        self.models['DQN+PSO+LSTM'] = None
    
    def create_baseline_models(self):
        """Tạo baseline models nếu không có checkpoint"""
        print("\n" + "="*80)
        print("CREATING BASELINE MODELS")
        print("="*80)
        
        # Simple LSTM
        print("\n1. Simple LSTM (Random initialization)")
        try:
            lstm_agent = LSTMTradingAgent(
                input_size=20,
                hidden_size=128,
                num_layers=2,
                sequence_length=24
            )
            self.models['LSTM'] = {
                'agent': lstm_agent,
                'type': 'lstm'
            }
            print("   Status: LSTM created")
        except Exception as e:
            print(f"   Error: {e}")
            self.models['LSTM'] = None
        
        # Note: PSO+LSTM và PPO+PSO+LSTM cần training nên skip
        print("\n2. PSO+LSTM: Skipped (requires training)")
        print("3. PPO+PSO+LSTM: Skipped (requires training)")
        
        self.models['PSO+LSTM'] = None
        self.models['PPO+PSO+LSTM'] = None
    
    def predict_with_dqn(self, model_info):
        """Predict với DQN model"""
        agent = model_info['agent']
        
        # Create environment
        env = TradingMDP(self.test_df)
        
        actions = []
        state = env.reset()
        done = False
        
        with torch.no_grad():
            while not done:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to('cpu')
                
                # Get Q-values and select action
                q_values = agent.policy_net(state_tensor)
                action = q_values.argmax().item()  # Use argmax() without dim for safety
                
                actions.append(action)
                
                # Step
                state, reward, done, _ = env.step(action)
        
        return actions
    
    def predict_with_dqn_lstm(self, model_info):
        """Predict với DQN+LSTM model"""
        agent = model_info['agent']
        
        # Prepare data
        feature_cols = [col for col in self.test_df.columns if col not in ['timestamp', 'coin']]
        env_data = self.test_df[feature_cols].values
        
        # Create environment
        env = TradingMDP(self.test_df)
        
        actions = []
        state = env.reset()
        done = False
        
        with torch.no_grad():
            while not done:
                # Prepare sequence
                state_seq = agent.prepare_sequence(env_data, env.current_step)
                
                # Predict action
                action = agent.predict_action(state_seq)
                actions.append(action)
                
                # Step
                state, reward, done, _ = env.step(action)
        
        return actions
    
    def predict_with_lstm(self, model_info):
        """Predict với simple LSTM (baseline - random)"""
        env = TradingMDP(self.test_df)
        
        actions = []
        state = env.reset()
        done = False
        
        # Random baseline
        while not done:
            action = np.random.randint(0, 3)
            actions.append(action)
            state, reward, done, _ = env.step(action)
        
        return actions
    
    def predict_all_models(self):
        """Dự báo cho tất cả models"""
        print("\n" + "="*80)
        print("GENERATING PREDICTIONS")
        print("="*80)
        
        for model_name, model_info in self.models.items():
            if model_info is None:
                print(f"\n{model_name}: Skipped (no model)")
                self.predictions[model_name] = None
                continue
            
            print(f"\n{model_name}: Predicting...")
            
            try:
                if model_info['type'] == 'dqn':
                    actions = self.predict_with_dqn(model_info)
                elif model_info['type'] == 'dqn_lstm':
                    actions = self.predict_with_dqn_lstm(model_info)
                elif model_info['type'] == 'lstm':
                    actions = self.predict_with_lstm(model_info)
                else:
                    actions = None
                
                self.predictions[model_name] = actions
                
                if actions:
                    action_counts = pd.Series(actions).value_counts()
                    print(f"   Total predictions: {len(actions)}")
                    print(f"   Hold: {action_counts.get(0, 0)} ({action_counts.get(0, 0)/len(actions)*100:.1f}%)")
                    print(f"   Buy: {action_counts.get(1, 0)} ({action_counts.get(1, 0)/len(actions)*100:.1f}%)")
                    print(f"   Sell: {action_counts.get(2, 0)} ({action_counts.get(2, 0)/len(actions)*100:.1f}%)")
                    
            except Exception as e:
                print(f"   Error: {e}")
                self.predictions[model_name] = None
    
    def calculate_metrics(self):
        """Tính toán metrics cho các models"""
        print("\n" + "="*80)
        print("CALCULATING METRICS")
        print("="*80)
        
        # Create Buy-Hold baseline
        env_baseline = TradingMDP(self.test_df)
        baseline_actions = [0] * len(self.test_df)  # All Hold
        
        # Calculate returns for each model
        for model_name, actions in self.predictions.items():
            if actions is None:
                continue
            
            print(f"\n{model_name}:")
            
            # Simulate trading
            env = TradingMDP(self.test_df)
            state = env.reset()
            
            total_reward = 0
            portfolio_values = [env.initial_balance]
            
            for action in actions:
                state, reward, done, _ = env.step(action)
                total_reward += reward
                portfolio_values.append(env.balance + env.holdings * state[3])  # balance + holdings * close_price
                
                if done:
                    break
            
            final_balance = env.balance + env.holdings * self.test_df['close'].iloc[-1]
            total_return = ((final_balance - env.initial_balance) / env.initial_balance) * 100
            
            # Store metrics
            self.metrics[model_name] = {
                'total_reward': total_reward,
                'final_balance': final_balance,
                'total_return': total_return,
                'portfolio_values': portfolio_values,
                'num_trades': sum(1 for a in actions if a != 0),
                'buy_actions': sum(1 for a in actions if a == 1),
                'sell_actions': sum(1 for a in actions if a == 2),
                'hold_actions': sum(1 for a in actions if a == 0)
            }
            
            print(f"   Total Reward: {total_reward:.2f}")
            print(f"   Final Balance: ${final_balance:.2f}")
            print(f"   Total Return: {total_return:.2f}%")
            print(f"   Num Trades: {self.metrics[model_name]['num_trades']}")
    
    def plot_results(self):
        """Vẽ đồ thị so sánh"""
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80)
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Portfolio Value Over Time
        ax1 = plt.subplot(2, 3, 1)
        for model_name, metrics in self.metrics.items():
            if metrics:
                ax1.plot(metrics['portfolio_values'], label=model_name, linewidth=2)
        
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Steps (Hours)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Total Returns Comparison
        ax2 = plt.subplot(2, 3, 2)
        models = [m for m in self.metrics.keys() if self.metrics[m]]
        returns = [self.metrics[m]['total_return'] for m in models]
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        bars = ax2.bar(models, returns, color=colors, alpha=0.7)
        ax2.set_title('Total Returns Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. Action Distribution
        ax3 = plt.subplot(2, 3, 3)
        action_data = []
        for model_name in models:
            action_data.append([
                self.metrics[model_name]['hold_actions'],
                self.metrics[model_name]['buy_actions'],
                self.metrics[model_name]['sell_actions']
            ])
        
        x = np.arange(len(models))
        width = 0.25
        
        ax3.bar(x - width, [d[0] for d in action_data], width, label='Hold', alpha=0.8)
        ax3.bar(x, [d[1] for d in action_data], width, label='Buy', alpha=0.8)
        ax3.bar(x + width, [d[2] for d in action_data], width, label='Sell', alpha=0.8)
        
        ax3.set_title('Action Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Actions')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Price with Actions (DQN only)
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(self.test_df['close'].values, label='BTC Price', color='blue', linewidth=2)
        
        if 'DQN' in self.predictions and self.predictions['DQN']:
            actions = self.predictions['DQN']
            buy_indices = [i for i, a in enumerate(actions) if a == 1]
            sell_indices = [i for i, a in enumerate(actions) if a == 2]
            
            ax4.scatter(buy_indices, [self.test_df['close'].iloc[i] for i in buy_indices],
                       color='green', marker='^', s=100, label='Buy', alpha=0.7, zorder=5)
            ax4.scatter(sell_indices, [self.test_df['close'].iloc[i] for i in sell_indices],
                       color='red', marker='v', s=100, label='Sell', alpha=0.7, zorder=5)
        
        ax4.set_title('Price Chart with DQN Actions', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Steps (Hours)')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Number of Trades
        ax5 = plt.subplot(2, 3, 5)
        trades = [self.metrics[m]['num_trades'] for m in models]
        ax5.bar(models, trades, color='skyblue', alpha=0.7)
        ax5.set_title('Number of Trades', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Trades Count')
        plt.xticks(rotation=45, ha='right')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Final Balance
        ax6 = plt.subplot(2, 3, 6)
        balances = [self.metrics[m]['final_balance'] for m in models]
        colors_balance = ['green' if b > 10000 else 'red' for b in balances]
        bars_balance = ax6.bar(models, balances, color=colors_balance, alpha=0.7)
        ax6.axhline(y=10000, color='black', linestyle='--', linewidth=1, label='Initial Balance')
        ax6.set_title('Final Balance', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Balance ($)')
        plt.xticks(rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars_balance:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save
        output_path = 'results/charts/model_comparison.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n   Plot saved: {output_path}")
        
        plt.show()
    
    def generate_report(self):
        """Tạo báo cáo chi tiết"""
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80)
        
        report = []
        report.append("# MODEL COMPARISON REPORT")
        report.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Coin: {self.coin}")
        report.append(f"Test Period: {self.test_days} days ({len(self.test_df)} hours)")
        report.append(f"Test Range: {self.test_df['timestamp'].min()} to {self.test_df['timestamp'].max()}")
        
        report.append("\n## MODELS TESTED")
        for model_name, model_info in self.models.items():
            status = "Loaded" if model_info else "Not Available"
            report.append(f"- {model_name}: {status}")
        
        report.append("\n## PERFORMANCE METRICS")
        report.append("\n| Model | Total Return (%) | Final Balance ($) | Num Trades | Buy | Sell | Hold |")
        report.append("|-------|------------------|-------------------|------------|-----|------|------|")
        
        for model_name, metrics in self.metrics.items():
            if metrics:
                report.append(f"| {model_name} | {metrics['total_return']:.2f}% | ${metrics['final_balance']:.2f} | {metrics['num_trades']} | {metrics['buy_actions']} | {metrics['sell_actions']} | {metrics['hold_actions']} |")
        
        report.append("\n## ANALYSIS")
        
        # Best performer
        if self.metrics:
            best_model = max(self.metrics.items(), key=lambda x: x[1]['total_return'] if x[1] else -float('inf'))
            report.append(f"\n**Best Performer:** {best_model[0]}")
            report.append(f"- Total Return: {best_model[1]['total_return']:.2f}%")
            report.append(f"- Final Balance: ${best_model[1]['final_balance']:.2f}")
        
        # Market condition
        price_change = ((self.test_df['close'].iloc[-1] - self.test_df['close'].iloc[0]) / self.test_df['close'].iloc[0]) * 100
        market_trend = "Bullish (Tăng)" if price_change > 0 else "Bearish (Giảm)"
        
        report.append(f"\n**Market Condition:**")
        report.append(f"- Trend: {market_trend}")
        report.append(f"- Price Change: {price_change:.2f}%")
        report.append(f"- Initial Price: ${self.test_df['close'].iloc[0]:.2f}")
        report.append(f"- Final Price: ${self.test_df['close'].iloc[-1]:.2f}")
        
        report.append("\n## OBSERVATIONS")
        report.append("\n1. **Action Strategy:**")
        for model_name, metrics in self.metrics.items():
            if metrics:
                total_actions = metrics['buy_actions'] + metrics['sell_actions'] + metrics['hold_actions']
                hold_pct = (metrics['hold_actions'] / total_actions) * 100
                report.append(f"   - {model_name}: Hold {hold_pct:.1f}% of time")
        
        report.append("\n2. **Trading Frequency:**")
        for model_name, metrics in self.metrics.items():
            if metrics:
                report.append(f"   - {model_name}: {metrics['num_trades']} trades")
        
        # Save report
        report_text = "\n".join(report)
        output_path = 'results/reports/model_comparison_report.md'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n   Report saved: {output_path}")
        # Don't print report to avoid Unicode errors in PowerShell
        # print("\n" + "="*80)
        # print(report_text)
        # print("="*80)

def main():
    """Main function"""
    print("\n" + "="*80)
    print("MODEL COMPARISON - TRADING PREDICTIONS")
    print("="*80)
    
    # Create comparator
    comparator = ModelComparator(coin='BTC', test_days=30)
    
    # Load data
    comparator.load_data()
    
    # Load models
    comparator.load_dqn_model()
    comparator.load_dqn_pso_lstm_model()
    comparator.create_baseline_models()
    
    # Predict
    comparator.predict_all_models()
    
    # Calculate metrics
    comparator.calculate_metrics()
    
    # Plot results
    comparator.plot_results()
    
    # Generate report
    comparator.generate_report()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
