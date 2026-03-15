"""
So Sánh 3 Models: LSTM, PSO+LSTM, và PPO+PSO+LSTM
Đánh giá trên 1 ngày cụ thể và tính metrics: MAE, MSE, RMSE, MAPE, R2
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import models
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from models.lstm_trading import LSTMTradingAgent
from models.pso_lstm_trading import PSOLSTMAgent
from models.mdp_trading import TradingEnvironment
from models.dqn_agent import DQNAgent


class ModelComparer:
    """
    Class để so sánh các models trading
    """
    
    def __init__(self, data_path: str, test_date: str = "2024-01-15"):
        """
        Args:
            data_path: Path đến file CSV chứa data
            test_date: Ngày để test (format: YYYY-MM-DD)
        """
        self.data_path = data_path
        self.test_date = test_date
        self.results_dir = Path(__file__).parent / 'results' / 'model_comparison'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("Loading data...")
        self.load_data()
        
        # Models
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
        # Action mapping
        self.action_names = {0: 'Buy', 1: 'Sell', 2: 'Hold'}
        
    def load_data(self):
        """
        Load và prepare data
        """
        # Load CSV (chỉ Bitcoin)
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter chỉ BTC
        df = df[df['coin'] == 'BTC'].copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Loaded {len(df)} records for BTC")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Features để dùng
        self.feature_columns = ['close', 'volume', 'rsi', 'macd', 'macd_signal', 
                               'macd_hist', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                               'bb_upper', 'bb_middle', 'bb_lower', 'adx', 'trend']
        
        # Tạo labels (actions) dựa trên price movement
        df['price_change'] = df['close'].pct_change()
        df['action'] = 2  # Default: Hold
        
        # Buy signal: Price sẽ tăng > 0.5%
        df.loc[df['price_change'].shift(-1) > 0.005, 'action'] = 0
        
        # Sell signal: Price sẽ giảm > 0.5%
        df.loc[df['price_change'].shift(-1) < -0.005, 'action'] = 1
        
        # Remove NaN
        df = df.dropna().reset_index(drop=True)
        
        # Split data
        # Train: 70%, Val: 15%, Test: 15%
        n_total = len(df)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.85)
        
        self.df_train = df.iloc[:n_train].copy()
        self.df_val = df.iloc[n_train:n_val].copy()
        self.df_test = df.iloc[n_val:].copy()
        
        print(f"\nData split:")
        print(f"  Train: {len(self.df_train)} samples ({self.df_train['timestamp'].min()} to {self.df_train['timestamp'].max()})")
        print(f"  Val: {len(self.df_val)} samples ({self.df_val['timestamp'].min()} to {self.df_val['timestamp'].max()})")
        print(f"  Test: {len(self.df_test)} samples ({self.df_test['timestamp'].min()} to {self.df_test['timestamp'].max()})")
        
        # Normalize features
        self.scaler = MinMaxScaler()
        
        # Fit on train
        self.X_train = self.scaler.fit_transform(self.df_train[self.feature_columns].values)
        self.y_train = self.df_train['action'].values
        
        # Transform val và test
        self.X_val = self.scaler.transform(self.df_val[self.feature_columns].values)
        self.y_val = self.df_val['action'].values
        
        self.X_test = self.scaler.transform(self.df_test[self.feature_columns].values)
        self.y_test = self.df_test['action'].values
        
        print(f"\nFeature shape: {self.X_train.shape}")
        print(f"Action distribution (train): {np.bincount(self.y_train.astype(int))}")
        
    def train_lstm(self, epochs: int = 50):
        """
        Train LSTM model
        """
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL")
        print("="*60)
        
        agent = LSTMTradingAgent(
            input_size=len(self.feature_columns),
            hidden_size=128,
            num_layers=2,
            sequence_length=24,
            learning_rate=0.001
        )
        
        # Prepare sequences
        X_train_seq, y_train_seq = agent.prepare_sequences(self.X_train, self.y_train)
        X_val_seq, y_val_seq = agent.prepare_sequences(self.X_val, self.y_val)
        
        # Train
        agent.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            epochs=epochs,
            batch_size=32,
            patience=10,
            verbose=True
        )
        
        self.models['LSTM'] = agent
        print("\n✓ LSTM training complete!")
        
        # Save
        model_path = self.results_dir / 'lstm_model.pth'
        agent.save(str(model_path))
        print(f"Model saved to {model_path}")
        
    def train_pso_lstm(self, n_particles: int = 6, max_iterations: int = 10):
        """
        Train PSO+LSTM model
        """
        print("\n" + "="*60)
        print("TRAINING PSO+LSTM MODEL")
        print("="*60)
        
        agent = PSOLSTMAgent(
            input_size=len(self.feature_columns)
        )
        
        # Optimize và train
        agent.optimize_and_train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            n_particles=n_particles,
            max_iterations=max_iterations,
            final_epochs=50,
            verbose=True
        )
        
        self.models['PSO+LSTM'] = agent
        print("\n✓ PSO+LSTM training complete!")
        
        # Save
        model_path = self.results_dir / 'pso_lstm_model.pth'
        agent.save(str(model_path))
        print(f"Model saved to {model_path}")
        
    def train_dqn_baseline(self, episodes: int = 100):
        """
        Train DQN làm baseline (model hiện có trong dự án)
        """
        print("\n" + "="*60)
        print("TRAINING DQN MODEL (Baseline)")
        print("="*60)
        
        # Create environment
        env = TradingEnvironment(
            data=self.df_train,
            initial_balance=10000,
            transaction_cost=0.001
        )
        
        # Create DQN agent
        state_size = env.observation_space.shape[0]
        agent = DQNAgent(
            state_size=state_size,
            action_size=3,
            learning_rate=0.001
        )
        
        # Train
        rewards_history = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action = agent.select_action(state)
                
                # Step
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train
                if len(agent.memory) > agent.batch_size:
                    agent.train()
                
                episode_reward += reward
                state = next_state
            
            rewards_history.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Episode {episode+1}/{episodes}: Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        self.models['DQN'] = agent
        print("\n✓ DQN training complete!")
        
    def evaluate_all_models(self):
        """
        Evaluate tất cả models trên test set
        """
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            if model_name == 'DQN':
                # DQN evaluation (qua environment)
                predictions = self._evaluate_dqn(model)
            else:
                # LSTM-based models
                predictions = self._evaluate_lstm_model(model)
            
            self.predictions[model_name] = predictions
            
            # Calculate metrics
            self.metrics[model_name] = self._calculate_metrics(predictions, self.y_test)
            
            print(f"✓ {model_name} evaluation complete!")
            print(f"  Metrics: {self.metrics[model_name]}")
    
    def _evaluate_lstm_model(self, model):
        """
        Evaluate LSTM-based model
        """
        predictions = []
        sequence_length = model.agent.sequence_length if hasattr(model, 'agent') else model.sequence_length
        
        for i in range(sequence_length, len(self.X_test)):
            sequence = self.X_test[i-sequence_length:i]
            action = model.predict_action(sequence)
            predictions.append(action)
        
        # Pad đầu với Hold
        predictions = [2] * sequence_length + predictions
        
        return np.array(predictions)
    
    def _evaluate_dqn(self, agent):
        """
        Evaluate DQN model qua environment
        """
        env = TradingEnvironment(
            data=self.df_test,
            initial_balance=10000,
            transaction_cost=0.001
        )
        
        state = env.reset()
        predictions = []
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon=0.0)  # Greedy
            predictions.append(action)
            
            state, _, done, _ = env.step(action)
        
        # Pad nếu cần
        while len(predictions) < len(self.y_test):
            predictions.append(2)  # Hold
        
        return np.array(predictions[:len(self.y_test)])
    
    def _calculate_metrics(self, predictions, actual):
        """
        Calculate regression metrics
        (Treat actions as continuous values để calculate MAE, MSE, etc.)
        """
        # Ensure same length
        min_len = min(len(predictions), len(actual))
        pred = predictions[:min_len]
        act = actual[:min_len]
        
        # Metrics
        mae = mean_absolute_error(act, pred)
        mse = mean_squared_error(act, pred)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = act != 0
        mape = np.mean(np.abs((act[mask] - pred[mask]) / act[mask])) * 100 if mask.sum() > 0 else 0
        
        # R2 score
        r2 = r2_score(act, pred)
        
        # Accuracy (classification metric)
        accuracy = (pred == act).mean() * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Accuracy': accuracy
        }
    
    def plot_comparison(self):
        """
        Vẽ đồ thị so sánh
        """
        print("\n" + "="*60)
        print("GENERATING COMPARISON PLOTS")
        print("="*60)
        
        # 1. Metrics comparison bar chart
        self._plot_metrics_comparison()
        
        # 2. Predictions timeline
        self._plot_predictions_timeline()
        
        # 3. Confusion matrices
        self._plot_confusion_matrices()
        
        print("\n✓ All plots generated!")
    
    def _plot_metrics_comparison(self):
        """
        Plot metrics comparison
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('So Sánh Metrics Giữa Các Models', fontsize=16, fontweight='bold')
        
        metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'Accuracy']
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx // 3, idx % 3]
            
            model_names = list(self.metrics.keys())
            values = [self.metrics[model][metric_name] for model in model_names]
            
            # Bar plot
            bars = ax.bar(model_names, values, color=['#3498db', '#e74c3c', '#2ecc71'])
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}' if abs(height) < 10 else f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = self.results_dir / 'metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def _plot_predictions_timeline(self):
        """
        Plot predictions over time
        """
        fig, axes = plt.subplots(len(self.models), 1, figsize=(15, 4*len(self.models)))
        
        if len(self.models) == 1:
            axes = [axes]
        
        timestamps = self.df_test['timestamp'].values[:len(self.y_test)]
        
        for idx, (model_name, predictions) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            # Plot actual
            ax.plot(timestamps, self.y_test, label='Actual', alpha=0.7, linewidth=2)
            
            # Plot predictions
            ax.plot(timestamps, predictions, label='Predicted', alpha=0.7, linewidth=2)
            
            ax.set_title(f'{model_name} - Predictions vs Actual', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Action (0=Buy, 1=Sell, 2=Hold)')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / 'predictions_timeline.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def _plot_confusion_matrices(self):
        """
        Plot confusion matrices
        """
        from sklearn.metrics import confusion_matrix
        
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, predictions) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            # Calculate confusion matrix
            cm = confusion_matrix(self.y_test, predictions)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Buy', 'Sell', 'Hold'],
                       yticklabels=['Buy', 'Sell', 'Hold'])
            ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        save_path = self.results_dir / 'confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def save_results_to_file(self):
        """
        Save results to CSV and text file
        """
        # Save metrics to CSV
        df_metrics = pd.DataFrame(self.metrics).T
        csv_path = self.results_dir / 'metrics_comparison.csv'
        df_metrics.to_csv(csv_path)
        print(f"\nSaved metrics to: {csv_path}")
        
        # Save detailed report
        report_path = self.results_dir / 'comparison_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BÁO CÁO SO SÁNH CÁC MODELS TRADING\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Ngày đánh giá: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test set: {len(self.y_test)} samples\n")
            f.write(f"Date range: {self.df_test['timestamp'].min()} to {self.df_test['timestamp'].max()}\n\n")
            
            f.write("="*80 + "\n")
            f.write("METRICS COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            f.write(df_metrics.to_string())
            f.write("\n\n")
            
            f.write("="*80 + "\n")
            f.write("RANKING (Best to Worst)\n")
            f.write("="*80 + "\n\n")
            
            for metric in ['Accuracy', 'R2', 'MAE', 'RMSE']:
                if metric in ['Accuracy', 'R2']:
                    # Higher is better
                    ranked = df_metrics.sort_values(metric, ascending=False)
                else:
                    # Lower is better
                    ranked = df_metrics.sort_values(metric, ascending=True)
                
                f.write(f"{metric}:\n")
                for i, (model, value) in enumerate(ranked[metric].items(), 1):
                    f.write(f"  {i}. {model}: {value:.4f}\n")
                f.write("\n")
        
        print(f"Saved detailed report to: {report_path}")


def main():
    """
    Main function
    """
    print("="*80)
    print("SO SÁNH CÁC MODELS TRADING: LSTM vs PSO+LSTM vs DQN")
    print("="*80)
    
    # Path to data
    data_path = Path(__file__).parent / 'data' / 'raw' / 'multi_coin_1h.csv'
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Create comparer
    comparer = ModelComparer(str(data_path))
    
    # Train models
    print("\n" + "="*80)
    print("PHASE 1: TRAINING MODELS")
    print("="*80)
    
    comparer.train_lstm(epochs=50)
    comparer.train_pso_lstm(n_particles=6, max_iterations=10)
    comparer.train_dqn_baseline(episodes=100)
    
    # Evaluate
    print("\n" + "="*80)
    print("PHASE 2: EVALUATION")
    print("="*80)
    
    comparer.evaluate_all_models()
    
    # Plot
    print("\n" + "="*80)
    print("PHASE 3: VISUALIZATION")
    print("="*80)
    
    comparer.plot_comparison()
    
    # Save
    comparer.save_results_to_file()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {comparer.results_dir}")


if __name__ == '__main__':
    main()
