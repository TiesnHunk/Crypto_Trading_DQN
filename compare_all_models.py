"""
So sánh tất cả các mô hình trading:
- LSTM
- PSO + LSTM
- PPO + PSO + LSTM
- DQN (baseline đã train)

Dự báo hành động cho 1 ngày cụ thể và so sánh kết quả
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.lstm_trading import LSTMTradingAgent
from src.models.pso_lstm_trading import PSOLSTMOptimizer
from src.models.ppo_pso_lstm_trading import PPOPSOOptimizer
from src.models.dqn_agent import DQNAgent
from src.models.dqn_pso_lstm_trading import DQNPSOLSTMOptimizer
from src.models.mdp_trading import TradingMDP
from src.utils.indicators import TechnicalIndicators


class ModelComparison:
    """
    Class để so sánh các mô hình trading
    """
    
    def __init__(self, data_path: str, test_date: str = None):
        """
        Args:
            data_path: Path đến file CSV
            test_date: Ngày cần dự báo (format: 'YYYY-MM-DD')
        """
        print("="*80)
        print("🔬 KHỞI TẠO SO SÁNH CÁC MÔ HÌNH TRADING")
        print("="*80)
        
        # Load data
        print(f"\n📥 Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"   ✅ Loaded {len(self.df):,} rows")
        print(f"   📅 Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        # Filter BTC only cho test
        self.df = self.df[self.df['coin'] == 'BTC'].copy()
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"   📊 BTC data: {len(self.df):,} rows")
        
        # Determine test date
        if test_date is None:
            # Use last available day
            test_date = self.df['timestamp'].max() - timedelta(days=1)
        else:
            test_date = pd.to_datetime(test_date)
        
        self.test_date = test_date
        print(f"\n🎯 Test Date: {test_date.strftime('%Y-%m-%d')}")
        
        # Split train/test
        self._split_data()
        
        # Results storage
        self.results = {}
        self.predictions = {}
        self.metrics = {}
        
    def _split_data(self):
        """
        Split dữ liệu thành train và test
        Test: 24 giờ của test_date
        Train: tất cả data trước test_date
        """
        # Test day: 24 hours của test_date
        test_start = self.test_date.replace(hour=0, minute=0, second=0)
        test_end = test_start + timedelta(days=1)
        
        # Split
        self.train_df = self.df[self.df['timestamp'] < test_start].copy()
        self.test_df = self.df[
            (self.df['timestamp'] >= test_start) & 
            (self.df['timestamp'] < test_end)
        ].copy()
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {len(self.train_df):,} samples ({self.train_df['timestamp'].min()} to {self.train_df['timestamp'].max()})")
        print(f"   Test: {len(self.test_df):,} samples (24 hours of {self.test_date.strftime('%Y-%m-%d')})")
        
        if len(self.test_df) < 24:
            print(f"   ⚠️ Warning: Test set has only {len(self.test_df)} hours")
    
    def prepare_features(self, df):
        """
        Chuẩn bị features cho model
        """
        features = ['open', 'high', 'low', 'close', 'volume',
                   'rsi', 'macd', 'macd_signal', 'macd_hist',
                   'sma_20', 'sma_50', 'ema_12', 'ema_26',
                   'bb_upper', 'bb_middle', 'bb_lower',
                   'adx', 'price_change', 'volatility']
        
        # Fill NaN
        data = df[features].copy()
        data = data.fillna(method='ffill').fillna(0)
        
        return data.values
    
    def generate_labels(self, df):
        """
        Generate labels từ price movement
        Buy=0, Sell=1, Hold=2
        """
        labels = []
        for i in range(len(df)):
            if i < len(df) - 1:
                price_change = (df.iloc[i+1]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
                
                if price_change > 0.001:  # >0.1% -> Buy
                    labels.append(0)
                elif price_change < -0.001:  # <-0.1% -> Sell
                    labels.append(1)
                else:
                    labels.append(2)  # Hold
            else:
                labels.append(2)  # Last sample -> Hold
        
        return np.array(labels)
    
    def train_lstm(self):
        """
        Train LSTM model
        """
        print("\n" + "="*80)
        print("🤖 TRAINING LSTM MODEL")
        print("="*80)
        
        # Prepare data
        X_train = self.prepare_features(self.train_df)
        y_train = self.generate_labels(self.train_df)
        
        print(f"   Features: {X_train.shape}")
        print(f"   Labels: {y_train.shape}")
        
        # Initialize agent
        agent = LSTMTradingAgent(
            input_size=X_train.shape[1],
            hidden_size=128,
            num_layers=2,
            sequence_length=24,
            learning_rate=0.001
        )
        
        # Train
        print(f"\n   Training for 50 epochs...")
        history = agent.train(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_split=0.2,
            verbose=True
        )
        
        self.lstm_agent = agent
        
        print(f"   ✅ LSTM training complete!")
        print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
        
        return agent
    
    def train_pso_lstm(self):
        """
        Train PSO + LSTM model
        """
        print("\n" + "="*80)
        print("🔬 TRAINING PSO + LSTM MODEL")
        print("="*80)
        
        # Prepare data
        X_train = self.prepare_features(self.train_df)
        y_train = self.generate_labels(self.train_df)
        
        # Initialize PSO optimizer
        optimizer = PSOLSTMOptimizer(
            n_particles=5,  # Giảm xuống để nhanh hơn
            max_iterations=10,
            w=0.7, c1=1.5, c2=1.5
        )
        
        # Optimize
        print(f"\n   Optimizing hyperparameters with PSO...")
        print(f"   Particles: 5, Iterations: 10")
        
        best_params, best_agent, history = optimizer.optimize(
            X_train, y_train,
            input_size=X_train.shape[1],
            validation_split=0.2,
            epochs_per_eval=20,
            verbose=True
        )
        
        self.pso_agent = best_agent
        
        print(f"\n   ✅ PSO + LSTM optimization complete!")
        print(f"   Best parameters: {best_params}")
        print(f"   Best fitness: {history['global_best_fitness'][-1]:.4f}")
        
        return best_agent
    
    def train_ppo_pso_lstm(self):
        """
        Train PPO + PSO + LSTM model
        """
        print("\n" + "="*80)
        print("🚀 TRAINING PPO + PSO + LSTM MODEL")
        print("="*80)
        
        # Create trading environment
        mdp = TradingMDP(
            data=self.train_df,
            initial_balance=10000,
            transaction_cost=0.0001
        )
        
        # Initialize PPO-PSO optimizer
        optimizer = PPOPSOOptimizer(
            n_particles=5,
            max_iterations=10,
            ppo_epochs=3,  # Giảm để training nhanh hơn
            w=0.7, c1=1.5, c2=1.5
        )
        
        print(f"\n   Optimizing with PPO + PSO...")
        print(f"   Particles: 5, Iterations: 10, PPO epochs: 3")
        
        # Optimize
        best_params, best_agent, history = optimizer.optimize(
            mdp=mdp,
            input_size=7,  # MDP state size
            n_episodes=50,  # Giảm episodes
            verbose=True
        )
        
        self.ppo_agent = best_agent
        
        print(f"\n   ✅ PPO + PSO + LSTM optimization complete!")
        print(f"   Best parameters: {best_params}")
        print(f"   Best fitness: {history['global_best_fitness'][-1]:.4f}")
        
        return best_agent
    
    def train_dqn_pso_lstm(self):
        """
        Train DQN + PSO + LSTM model
        """
        print("\n" + "="*80)
        print("⚡ TRAINING DQN + PSO + LSTM MODEL")
        print("="*80)
        
        # Create trading environment
        mdp = TradingMDP(
            data=self.train_df,
            initial_balance=10000,
            transaction_cost=0.0001
        )
        
        # Prepare features
        X_train = self.prepare_features(self.train_df)
        
        # Initialize DQN-PSO optimizer
        optimizer = DQNPSOLSTMOptimizer(
            n_particles=5,
            max_iterations=10,
            w=0.7, c1=1.5, c2=1.5
        )
        
        print(f"\n   Optimizing with DQN + PSO + LSTM...")
        print(f"   Particles: 5, Iterations: 10")
        
        # Optimize
        best_params, best_agent, history = optimizer.optimize(
            env=mdp,
            input_size=X_train.shape[1],
            verbose=True
        )
        
        self.dqn_pso_agent = best_agent
        
        print(f"\n   ✅ DQN + PSO + LSTM optimization complete!")
        print(f"   Best parameters: {best_params}")
        print(f"   Best fitness: {history['global_best_fitness'][-1]:.4f}")
        
        return best_agent
    
    def load_dqn(self):
        """
        Load DQN model đã train
        """
        print("\n" + "="*80)
        print("📦 LOADING DQN MODEL")
        print("="*80)
        
        checkpoint_path = Path('src/checkpoints_dqn/checkpoint_best.pkl')
        
        if not checkpoint_path.exists():
            print(f"   ❌ DQN checkpoint not found: {checkpoint_path}")
            return None
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Create agent
        agent = DQNAgent(
            state_dim=checkpoint.get('state_dim', 7),
            action_dim=checkpoint.get('action_dim', 3),
            hidden_dim_1=checkpoint.get('hidden_dim_1', 128),
            hidden_dim_2=checkpoint.get('hidden_dim_2', 64),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Load weights
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        
        self.dqn_agent = agent
        
        print(f"   ✅ DQN model loaded successfully!")
        print(f"   Episodes trained: {checkpoint.get('episode', 'unknown')}")
        print(f"   Best reward: {checkpoint.get('best_reward', 'unknown')}")
        
        return agent
    
    def predict_lstm(self, X_test):
        """
        Dự báo với LSTM
        """
        return self.lstm_agent.predict(X_test)
    
    def predict_pso_lstm(self, X_test):
        """
        Dự báo với PSO + LSTM
        """
        return self.pso_agent.predict(X_test)
    
    def predict_ppo_lstm(self, test_df):
        """
        Dự báo với PPO + PSO + LSTM
        """
        # Create test environment
        mdp = TradingMDP(
            data=test_df,
            initial_balance=10000,
            transaction_cost=0.0001
        )
        
        actions = []
        mdp.reset()
        
        for i in range(len(test_df)):
            state = mdp.get_state()
            action = self.ppo_agent.select_action(state, training=False)
            actions.append(action)
            
            if i < len(test_df) - 1:
                _, _, done, _ = mdp.step(action)
                if done:
                    break
        
        return np.array(actions)
    
    def predict_dqn(self, test_df):
        """
        Dự báo với DQN
        """
        # Create test environment
        mdp = TradingMDP(
            data=test_df,
            initial_balance=10000,
            transaction_cost=0.0001
        )
        
        actions = []
        mdp.reset()
        
        for i in range(len(test_df)):
            state = mdp.get_state()
            action = self.dqn_agent.select_action(state, epsilon=0)  # Greedy
            actions.append(action)
            
            if i < len(test_df) - 1:
                _, _, done, _ = mdp.step(action)
                if done:
                    break
        
        return np.array(actions)
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """
        Tính toán metrics
        """
        # Convert actions to prices for regression metrics
        # Assumption: Buy -> price will go up, Sell -> price will go down
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
        
        # R2 score
        r2 = r2_score(y_true, y_pred)
        
        # Accuracy (for action classification)
        accuracy = np.mean(y_true == y_pred) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Accuracy': accuracy
        }
        
        self.metrics[model_name] = metrics
        
        return metrics
    
    def predict_dqn_pso_lstm(self, test_df):
        """
        Dự báo với DQN + PSO + LSTM
        """
        # Create test environment
        mdp = TradingMDP(
            data=test_df,
            initial_balance=10000,
            transaction_cost=0.0001
        )
        
        # Prepare features
        env_data = self.prepare_features(test_df)
        
        actions = []
        mdp.reset()
        
        for i in range(len(test_df)):
            # Get state sequence
            state_seq = self.dqn_pso_agent.prepare_sequence(env_data, mdp.current_step)
            
            # Predict action
            action = self.dqn_pso_agent.predict_action(state_seq)
            actions.append(action)
            
            if i < len(test_df) - 1:
                _, _, done, _ = mdp.step(action)
                if done:
                    break
        
        return np.array(actions)
    
    def run_comparison(self):
        """
        Chạy so sánh tất cả các mô hình
        """
        print("\n" + "="*80)
        print("🔄 BẮT ĐẦU SO SÁNH CÁC MÔ HÌNH")
        print("="*80)
        
        # 1. Train LSTM
        self.train_lstm()
        
        # 2. Train PSO + LSTM
        self.train_pso_lstm()
        
        # 3. Train PPO + PSO + LSTM
        self.train_ppo_pso_lstm()
        
        # 4. Train DQN + PSO + LSTM
        self.train_dqn_pso_lstm()
        
        # 5. Load DQN (baseline)
        self.load_dqn()
        
        # Prepare test data
        X_test = self.prepare_features(self.test_df)
        y_true = self.generate_labels(self.test_df)
        
        print("\n" + "="*80)
        print("📊 PREDICTIONS")
        print("="*80)
        
        # Predictions
        print("\n   Making predictions...")
        
        models = {}
        
        if hasattr(self, 'lstm_agent'):
            lstm_pred = self.predict_lstm(X_test)
            models['LSTM'] = lstm_pred
            print(f"   ✅ LSTM: {len(lstm_pred)} predictions")
        
        if hasattr(self, 'pso_agent'):
            pso_pred = self.predict_pso_lstm(X_test)
            models['PSO+LSTM'] = pso_pred
            print(f"   ✅ PSO+LSTM: {len(pso_pred)} predictions")
        
        if hasattr(self, 'ppo_agent'):
            ppo_pred = self.predict_ppo_lstm(self.test_df)
            models['PPO+PSO+LSTM'] = ppo_pred
            print(f"   ✅ PPO+PSO+LSTM: {len(ppo_pred)} predictions")
        
        if hasattr(self, 'dqn_pso_agent'):
            dqn_pso_pred = self.predict_dqn_pso_lstm(self.test_df)
            models['DQN+PSO+LSTM'] = dqn_pso_pred
            print(f"   ✅ DQN+PSO+LSTM: {len(dqn_pso_pred)} predictions")
        
        if hasattr(self, 'dqn_agent'):
            dqn_pred = self.predict_dqn(self.test_df)
            models['DQN'] = dqn_pred
            print(f"   ✅ DQN: {len(dqn_pred)} predictions")
        
        # Calculate metrics
        print("\n" + "="*80)
        print("📈 METRICS")
        print("="*80)
        
        for model_name, predictions in models.items():
            # Ensure same length
            min_len = min(len(y_true), len(predictions))
            metrics = self.calculate_metrics(
                y_true[:min_len], 
                predictions[:min_len],
                model_name
            )
            
            print(f"\n   {model_name}:")
            for metric, value in metrics.items():
                print(f"      {metric}: {value:.4f}")
        
        # Store results
        self.predictions = models
        
        return models, self.metrics
    
    def plot_results(self):
        """
        Vẽ đồ thị so sánh
        """
        print("\n" + "="*80)
        print("📊 GENERATING PLOTS")
        print("="*80)
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Actions over time
        ax1 = plt.subplot(3, 2, 1)
        times = self.test_df['timestamp'].values[:len(list(self.predictions.values())[0])]
        
        for model_name, actions in self.predictions.items():
            ax1.plot(times, actions, label=model_name, marker='o', alpha=0.7)
        
        ax1.set_title('Trading Actions Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Action (0=Buy, 1=Sell, 2=Hold)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Price chart with actions
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(self.test_df['timestamp'], self.test_df['close'], 
                label='BTC Price', color='black', linewidth=2)
        ax2.set_title('BTC Price on Test Day', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. Metrics comparison - Bar chart
        ax3 = plt.subplot(3, 2, 3)
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df[['MAE', 'RMSE']].plot(kind='bar', ax=ax3)
        ax3.set_title('Error Metrics Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 4. R2 and Accuracy
        ax4 = plt.subplot(3, 2, 4)
        metrics_df[['R2', 'Accuracy']].plot(kind='bar', ax=ax4)
        ax4.set_title('R² and Accuracy Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 5. Action distribution
        ax5 = plt.subplot(3, 2, 5)
        action_counts = {}
        for model_name, actions in self.predictions.items():
            unique, counts = np.unique(actions, return_counts=True)
            action_counts[model_name] = dict(zip(unique, counts))
        
        action_df = pd.DataFrame(action_counts).fillna(0)
        action_df.plot(kind='bar', ax=ax5)
        ax5.set_title('Action Distribution by Model', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Action (0=Buy, 1=Sell, 2=Hold)')
        ax5.set_ylabel('Count')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.xticks(rotation=0)
        
        # 6. MAPE comparison
        ax6 = plt.subplot(3, 2, 6)
        mape_data = {k: v['MAPE'] for k, v in self.metrics.items()}
        plt.bar(mape_data.keys(), mape_data.values(), color=['blue', 'orange', 'green', 'red'])
        ax6.set_title('MAPE Comparison', fontsize=14, fontweight='bold')
        ax6.set_ylabel('MAPE (%)')
        ax6.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save
        output_path = Path('results/charts/model_comparison.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"   ✅ Chart saved to: {output_path}")
        
        plt.show()
    
    def generate_report(self):
        """
        Tạo báo cáo chi tiết
        """
        print("\n" + "="*80)
        print("📝 GENERATING REPORT")
        print("="*80)
        
        report = []
        report.append("# BÁO CÁO SO SÁNH CÁC MÔ HÌNH TRADING\n")
        report.append("="*80 + "\n\n")
        
        # 1. Thông tin test
        report.append("## 1. Thông Tin Test\n\n")
        report.append(f"- **Test Date**: {self.test_date.strftime('%Y-%m-%d')}\n")
        report.append(f"- **Số samples**: {len(self.test_df)}\n")
        report.append(f"- **Cryptocurrency**: BTC\n")
        report.append(f"- **Price range**: ${self.test_df['close'].min():.2f} - ${self.test_df['close'].max():.2f}\n\n")
        
        # 2. Metrics
        report.append("## 2. Metrics So Sánh\n\n")
        report.append("| Model | MAE | MSE | RMSE | MAPE (%) | R² | Accuracy (%) |\n")
        report.append("|-------|-----|-----|------|----------|----|--------------|\n")
        
        for model_name, metrics in self.metrics.items():
            report.append(f"| {model_name} | {metrics['MAE']:.4f} | {metrics['MSE']:.4f} | "
                         f"{metrics['RMSE']:.4f} | {metrics['MAPE']:.2f} | "
                         f"{metrics['R2']:.4f} | {metrics['Accuracy']:.2f} |\n")
        
        report.append("\n")
        
        # 3. Action distribution
        report.append("## 3. Phân Bố Hành Động\n\n")
        for model_name, actions in self.predictions.items():
            unique, counts = np.unique(actions, return_counts=True)
            action_dict = dict(zip(unique, counts))
            
            buy_count = action_dict.get(0, 0)
            sell_count = action_dict.get(1, 0)
            hold_count = action_dict.get(2, 0)
            
            report.append(f"**{model_name}**:\n")
            report.append(f"- Buy: {buy_count} ({buy_count/len(actions)*100:.1f}%)\n")
            report.append(f"- Sell: {sell_count} ({sell_count/len(actions)*100:.1f}%)\n")
            report.append(f"- Hold: {hold_count} ({hold_count/len(actions)*100:.1f}%)\n\n")
        
        # Save report
        report_path = Path('results/reports/model_comparison_report.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"   ✅ Report saved to: {report_path}")
        
        return ''.join(report)


def main():
    """
    Main function
    """
    print("\n" + "="*80)
    print("🚀 MODEL COMPARISON SCRIPT")
    print("="*80)
    
    # Configuration
    data_path = 'data/raw/multi_coin_1h.csv'
    test_date = '2025-10-31'  # Có thể thay đổi
    
    # Run comparison
    comparison = ModelComparison(data_path, test_date)
    models, metrics = comparison.run_comparison()
    
    # Plot results
    comparison.plot_results()
    
    # Generate report
    report = comparison.generate_report()
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)
    print("\nKết quả đã được lưu vào:")
    print("   - Đồ thị: results/charts/model_comparison.png")
    print("   - Báo cáo: results/reports/model_comparison_report.md")
    

if __name__ == "__main__":
    main()
