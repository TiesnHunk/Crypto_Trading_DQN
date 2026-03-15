"""
Training script với Multi-Coin Data
Train Q-Learning agent trên nhiều cryptocurrencies
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime
import io

# ✅ Fix UnicodeEncodeError on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ✅ FIX OUTPUT BUFFERING: Force unbuffered output để hiển thị real-time
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    # Fallback: Use unbuffered mode if reconfigure not available
    import os
    os.environ['PYTHONUNBUFFERED'] = '1'

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.mdp_trading import TradingMDP
from src.models.q_learning_gpu import QLearningAgent
from src.utils.checkpoint import TrainingCheckpoint
from src.utils.indicators import TechnicalIndicators  # ✅ Import indicators
from src.config import config
from src.config.config import DEVICE, USE_GPU  # ✅ Import GPU config


class MultiCoinTrainer:
    """
    Trainer cho multi-coin Q-Learning
    """
    
    def __init__(self):
        """
        Initialize trainer
        """
        checkpoint_dir = Path(__file__).parent / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_manager = TrainingCheckpoint(str(checkpoint_dir))
        
    def load_multi_coin_data(self, filepath: str = None, data_split: str = None) -> pd.DataFrame:
        """
        Load multi-coin data với TIME-SERIES SPLIT strategy
        
        ✅ NEW STRATEGY: Load tất cả 5 coins từ multi_coin_1h.csv
        Training và Validation sẽ được split theo thời gian sau đó
        
        Args:
            filepath: Path to multi-coin CSV file (if None, auto-select multi_coin_1h.csv)
            data_split: DEPRECATED - Không còn dùng, giữ lại để tương thích
            
        Returns:
            DataFrame với tất cả 5 coins (BTC, ETH, BNB, SOL, ADA)
        """
        if filepath is None:
            # ✅ TIME-SERIES STRATEGY: Load tất cả 5 coins từ multi_coin_1h.csv
            filepath = Path(__file__).parent.parent / 'data' / 'raw' / 'multi_coin_1h.csv'
            if not filepath.exists():
                filepath = Path('data') / 'raw' / 'multi_coin_1h.csv'
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            print("   Please run: python src/data/prepare_multi_coin_data.py")
            return None
        
        print(f"\n📥 Loading multi-coin data from {filepath}...")
        # ✅ Fix: Thêm low_memory=False để tránh DtypeWarning
        df = pd.read_csv(filepath, low_memory=False)
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        print(f"✅ Loaded {len(df):,} rows")
        print(f"   Coins: {df['coin'].unique()}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        
        # ✅ Thêm technical indicators nếu chưa có
        required_cols = ['rsi', 'macd_hist', 'trend', 'bb_upper', 'bb_lower', 'volatility']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\n📊 Adding technical indicators (missing: {missing_cols})...")
            df = TechnicalIndicators.add_all_indicators(df)
            
            # ✅ FIX: Fill NaN values để đảm bảo state features đa dạng
            df['rsi'] = df['rsi'].fillna(50.0)  # Neutral RSI = 50
            df['macd_hist'] = df['macd_hist'].fillna(0.0)
            df['volatility'] = df['volatility'].fillna(0.0)
            
            df = df.dropna()
            print(f"✅ Indicators added. Remaining rows: {len(df):,}")
            if 'rsi' in df.columns:
                print(f"   RSI range: {df['rsi'].min():.2f} - {df['rsi'].max():.2f}")
        
        return df
    
    def split_by_coin(self, df: pd.DataFrame) -> dict:
        """
        Split data theo từng coin
        
        Returns:
            Dict {coin_name: DataFrame}
        """
        if 'coin' not in df.columns:
            return {'ALL': df}
        
        coins_data = {}
        for coin in df['coin'].unique():
            coin_df = df[df['coin'] == coin].copy()
            
            # ✅ Remove 'coin' column (không cần cho MDP)
            if 'coin' in coin_df.columns:
                coin_df = coin_df.drop('coin', axis=1)
            
            # ✅ Đảm bảo index là timestamp (nếu chưa có)
            if not isinstance(coin_df.index, pd.DatetimeIndex):
                if 'timestamp' in coin_df.columns:
                    coin_df = coin_df.set_index('timestamp')
                else:
                    coin_df.index = pd.date_range(start='2018-01-01', periods=len(coin_df), freq='1h')
            
            coins_data[coin] = coin_df
            print(f"   {coin}: {len(coins_data[coin]):,} samples")
        
        return coins_data
    
    def split_by_time(self, df: pd.DataFrame, 
                     train_start: str = '2020-01-01',
                     train_end: str = '2023-12-31',
                     val_start: str = '2024-01-01',
                     val_end: str = '2024-06-30',
                     train_ratio: float = None) -> tuple:
        """
        ✅ TIME-SERIES VALIDATION: Split data theo thời gian với tất cả 5 coins
        Training: Dữ liệu từ train_start đến train_end (tất cả coins)
        Validation: Dữ liệu từ val_start đến val_end (tất cả coins)
        
        Lợi ích:
        - Mô hình học từ tất cả 5 coins trong quá khứ
        - Validation trên giai đoạn tương lai, phản ánh khả năng sinh lời thực tế
        - Tránh validation bias do sự kiện vĩ mô tại cùng thời điểm
        
        Args:
            df: Multi-coin DataFrame với timestamp index
            train_start: Ngày bắt đầu training (default: '2020-01-01')
            train_end: Ngày kết thúc training (default: '2023-12-31')
            val_start: Ngày bắt đầu validation (default: '2024-01-01')
            val_end: Ngày kết thúc validation (default: '2024-06-30')
            train_ratio: DEPRECATED - Nếu None, dùng train_start/train_end. Nếu có, dùng ratio cũ
        
        Returns:
            (train_df, validation_df) - Cả hai đều chứa tất cả 5 coins
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            # Convert to DatetimeIndex nếu chưa phải
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have timestamp index or 'timestamp' column")
        
        # Sort by timestamp để đảm bảo thứ tự thời gian
        df = df.sort_index()
        
        # Convert date strings to datetime
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        val_start_dt = pd.to_datetime(val_start)
        val_end_dt = pd.to_datetime(val_end)
        
        # Fallback to ratio-based split nếu train_ratio được cung cấp
        if train_ratio is not None:
            split_idx = int(len(df) * train_ratio)
            split_date = df.index[split_idx]
            train_df = df.iloc[:split_idx].copy()
            validation_df = df.iloc[split_idx:].copy()
        else:
            # Split theo ngày cụ thể
            train_df = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)].copy()
            validation_df = df[(df.index >= val_start_dt) & (df.index <= val_end_dt)].copy()
            split_date = train_end_dt
        
        train_start_actual = train_df.index.min()
        train_end_actual = train_df.index.max()
        val_start_actual = validation_df.index.min()
        val_end_actual = validation_df.index.max()
        
        # Show coin distribution
        train_coins = train_df['coin'].value_counts() if 'coin' in train_df.columns else {}
        val_coins = validation_df['coin'].value_counts() if 'coin' in validation_df.columns else {}
        
        print(f"\n{'='*70}")
        print(f"📅 TIME-SERIES VALIDATION SPLIT (TẤT CẢ 5 COINS)")
        print(f"{'='*70}")
        print(f"   Training Period:   {train_start_actual} to {train_end_actual}")
        print(f"   Training Samples:  {len(train_df):,}")
        if train_coins is not None and len(train_coins) > 0:
            print(f"   Training Coins:    {dict(train_coins)}")
        print(f"   Validation Period: {val_start_actual} to {val_end_actual}")
        print(f"   Validation Samples: {len(validation_df):,}")
        if val_coins is not None and len(val_coins) > 0:
            print(f"   Validation Coins:  {dict(val_coins)}")
        print(f"   Split Date: {split_date}")
        print(f"{'='*70}\n")
        
        return train_df, validation_df
    
    def train_multi_coin(self, df: pd.DataFrame, 
                        n_episodes: int = 5000,
                        mode: str = 'sequential',
                        use_dqn: bool = False,
                        resume: bool = True,
                        train_start: str = '2020-01-01',
                        train_end: str = '2023-12-31',
                        val_start: str = '2024-01-01',
                        val_end: str = '2024-06-30',
                        train_ratio: float = None) -> QLearningAgent:
        """
        Train Q-Learning agent trên multi-coin data với Time-Series Validation
        
        ✅ NEW STRATEGY: Training và Validation đều chứa TẤT CẢ 5 COINS
        Training: 2020-2023 (tất cả coins)
        Validation: 2024 (tất cả coins)
        
        Args:
            df: Multi-coin DataFrame (tất cả 5 coins)
            n_episodes: Số episodes training
            mode: 'sequential' (train từng coin) hoặc 'mixed' (train random)
            use_dqn: True để dùng Deep Q-Network (GPU), False để dùng Tabular (CPU)
            resume: True để resume từ checkpoint nếu có (default: True)
            train_start: Ngày bắt đầu training (default: '2020-01-01')
            train_end: Ngày kết thúc training (default: '2023-12-31')
            val_start: Ngày bắt đầu validation (default: '2024-01-01')
            val_end: Ngày kết thúc validation (default: '2024-06-30')
            train_ratio: DEPRECATED - Dùng train_start/train_end thay thế
            
        Returns:
            Trained agent
        """
        print("\n" + "="*70)
        print(f"🚀 MULTI-COIN Q-LEARNING TRAINING (TIME-SERIES SPLIT)")
        print(f"   Mode: {mode}")
        print(f"   Episodes: {n_episodes}")
        print(f"   Strategy: Tất cả 5 coins, split theo thời gian")
        print("="*70)
        
        # GPU Info
        print(f"\n💻 GPU Status:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Device: {DEVICE}")
        print(f"   Use DQN (GPU): {use_dqn}")
        
        # ✅ TIME-SERIES VALIDATION: Split data theo time với tất cả 5 coins
        train_df, validation_df = self.split_by_time(
            df, 
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            train_ratio=train_ratio
        )
        
        # ✅ V4: Resume từ checkpoint nếu có
        start_episode = 0
        agent = None
        checkpoint_data = None
        
        if resume:
            checkpoint_data = self.checkpoint_manager.load_checkpoint('checkpoint_latest.pkl')
            if checkpoint_data:
                print(f"\n🔄 Found checkpoint! Resuming training...")
                print(f"   Episode: {checkpoint_data['episode']}")
                print(f"   Timestamp: {checkpoint_data['timestamp']}")
                
                # Check if mode matches
                metadata = checkpoint_data.get('metadata', {})
                saved_mode = metadata.get('mode', 'sequential')
                if saved_mode != mode:
                    print(f"   ⚠️ Warning: Checkpoint mode '{saved_mode}' != current mode '{mode}'")
                    print(f"   Continue anyway...")
                
                # Load agent
                loaded_Q = checkpoint_data.get('Q')
                if loaded_Q:
                    # Agent is saved in checkpoint
                    if isinstance(loaded_Q, QLearningAgent):
                        agent = loaded_Q
                        print(f"   ✅ Loaded agent from checkpoint")
                        print(f"   Current Epsilon: {agent.epsilon:.4f} (will continue decaying)")
                        if hasattr(agent, 'Q') and isinstance(agent.Q, dict):
                            print(f"   Q-table states: {len(agent.Q):,}")
                        elif hasattr(agent, 'update_counter'):
                            print(f"   GPU Updates: {agent.update_counter:,}")
                    else:
                        print(f"   ⚠️ Unexpected agent type in checkpoint, creating new agent")
                
                start_episode = checkpoint_data['episode']
                print(f"   📍 Resuming from episode {start_episode + 1}/{n_episodes}")
        
        # Initialize agent nếu chưa load được
        if agent is None:
            print(f"\n🆕 Creating new agent...")
            # ✅ V3: Import risk management params
            from src.config.config import RISK_MANAGEMENT_PARAMS, Q_LEARNING_PARAMS
            
            # Initialize agent với GPU support và epsilon_decay mới
            agent = QLearningAgent(
                state_dim=8,  # ✅ NEW APPROACH: Updated from 6 to 8 (added current_profit, time_since_trade)
                n_actions=3,
                n_bins=10,
                alpha=0.1,
                gamma=0.95,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=Q_LEARNING_PARAMS['epsilon_decay'],  # ✅ V3: 0.9995 (chậm hơn)
                discrete=not use_dqn,  # False nếu dùng DQN (GPU)
                use_gpu=USE_GPU and use_dqn,  # Chỉ dùng GPU nếu DQN
                device=DEVICE
            )
        
        # ✅ TIME-SERIES VALIDATION: Chỉ split training data by coin (không dùng validation_df)
        coins_data = self.split_by_coin(train_df)
        
        print(f"\n📊 Training on {len(coins_data)} coins (Training Period Only):")
        for coin, coin_df in coins_data.items():
            print(f"   {coin}: {len(coin_df):,} samples")
        
        # ✅ Lưu validation_df để validate sau
        self.validation_df = validation_df
        print(f"\n💾 Validation data saved for later validation:")
        print(f"   Validation period: {validation_df.index.min()} to {validation_df.index.max()}")
        print(f"   Validation samples: {len(validation_df):,}")
        
        # Training
        if mode == 'sequential':
            agent = self._train_sequential(agent, coins_data, n_episodes, start_episode, checkpoint_data)
        elif mode == 'mixed':
            agent = self._train_mixed(agent, coins_data, n_episodes, start_episode, checkpoint_data)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return agent
    
    def _train_sequential(self, agent: QLearningAgent,
                         coins_data: dict,
                         n_episodes: int,
                         start_episode: int = 0,
                         checkpoint_data: dict = None) -> QLearningAgent:
        """
        Train sequential: mỗi episode chọn 1 coin random
        
        Args:
            agent: Q-Learning agent
            coins_data: Dict of coin dataframes
            n_episodes: Total episodes to train
            start_episode: Episode to start from (for resume)
            checkpoint_data: Checkpoint data for resume
        """
        print("\n🔄 Sequential Training Mode")
        print("   Each episode trains on 1 random coin")
        
        coin_names = list(coins_data.keys())
        episodes_per_save = 10  # ✅ Lưu mỗi 10 episodes
        keep_last_n_checkpoints = 10  # ✅ Giữ lại 10 checkpoints gần nhất (100 episodes)
        
        # ✅ Timing tracking
        start_time = time.time()
        episode_times = []
        
        # ✅ Track best profit và total episodes cho checkpoint - Resume từ checkpoint nếu có
        if checkpoint_data and checkpoint_data.get('history'):
            history = checkpoint_data['history']
            best_profit = history.get('best_profit', float('-inf'))
            all_episodes = history.get('episodes', [])
            all_profits = history.get('rewards', [])  # rewards là profits
            best_episode_num = history.get('best_episode', 0)
            print(f"   📊 Resumed tracking: {len(all_episodes)} episodes, best profit: ${best_profit:.2f}")
        else:
            best_profit = float('-inf')
            best_episode_num = 0
            all_episodes = []
            all_profits = []
        
        # ✅ V4: Track early stops để summary
        early_stop_count = 0
        early_stop_reasons = {}
        
        # ✅ DISABLE EARLY STOPPING: Set patience rất cao để model chạy hết episodes
        early_stopping_patience = 999999  # ✅ Disable early stopping - model sẽ chạy hết 5000 episodes
        best_profit_so_far = best_profit
        best_profit_so_far_episode = best_episode_num
        no_improvement_count = 0
        last_check_episode = start_episode
        
        for episode in range(start_episode, n_episodes):
            episode_start = time.time()
            
            # Random chọn 1 coin
            coin = np.random.choice(coin_names)
            coin_df = coins_data[coin]
            
            # ✅ MAJOR FIX: Create MDP với Risk Management DISABLED
            from src.config.config import RISK_MANAGEMENT_PARAMS
            mdp = TradingMDP(
                coin_df,
                initial_balance=10000.0,
                transaction_cost=0.001,
                hold_penalty=0.001,  # ✅ RETRAIN: Tăng từ 0.0001 → 0.001 (penalize Hold nhiều hơn, khuyến khích trading)
                interval='1h',
                stop_loss_pct=0.15,  # ✅ FIX EPISODE LENGTH: Tăng từ 0.10 → 0.15 (15%, ít force sell hơn)
                max_position_pct=RISK_MANAGEMENT_PARAMS['max_position_pct'],
                max_loss_pct=RISK_MANAGEMENT_PARAMS['max_loss_pct'],
                trailing_stop_pct=RISK_MANAGEMENT_PARAMS['trailing_stop_pct'],
                enable_risk_management=False  # ✅ MAJOR FIX: Disable early stopping để episodes chạy đủ dài
            )
            
            # Train 1 episode
            try:
                state = mdp.reset()
                done = False
                episode_reward = 0
                steps = 0
                
                last_info = None  # ✅ Track info từ step cuối cùng
                max_steps = len(coin_df)  # Limit để tránh infinite loop
                while not done and steps < max_steps:
                    # Get action
                    action = agent.select_action(state)
                    
                    # Take step
                    next_state, reward, done, step_info = mdp.step(action)
                    last_info = step_info  # ✅ Lưu info từ step cuối cùng
                    
                    # Update Q-table
                    agent.update_q_value(state, action, reward, next_state, done)
                    
                    # ✅ V4: Force mini-batch update mỗi 20 steps để GPU không idle
                    # (Đặc biệt quan trọng khi episodes ngắn)
                    if hasattr(agent, 'batch_buffer') and steps % 20 == 0 and len(agent.batch_buffer) > 0:
                        if hasattr(agent, 'min_batch_size'):
                            if len(agent.batch_buffer) >= agent.min_batch_size:
                                agent._update_batch()
                                agent.update_counter += 1
                    
                    state = next_state
                    episode_reward += reward
                    steps += 1
                    
            except Exception as e:
                print(f"⚠️ Error in episode {episode + 1} with {coin}: {e}")
                continue
            
            # ✅ Flush batch buffer nếu dùng DQN (đảm bảo tất cả samples được update)
            if hasattr(agent, 'flush_batch'):
                agent.flush_batch()
            
            # ✅ V4: Track early stops để summary (không spam log)
            if last_info and last_info.get('episode_ended_early', False):
                early_stop_reason = last_info.get('early_stop_reason')
                if early_stop_reason:
                    early_stop_count += 1
                    # Extract loss percentage từ reason
                    if 'Max loss exceeded:' in early_stop_reason:
                        try:
                            loss_pct = float(early_stop_reason.split(':')[1].strip().replace('%', ''))
                            loss_range = f"{int(loss_pct//10)*10}-{int(loss_pct//10)*10+10}%"
                            early_stop_reasons[loss_range] = early_stop_reasons.get(loss_range, 0) + 1
                        except:
                            pass
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # ✅ Tính portfolio value và profit
            portfolio_value = mdp.get_portfolio_value()
            profit = ((portfolio_value - 10000.0) / 10000.0) * 100
            profit_amount = portfolio_value - 10000.0  # Profit in USD
            
            # ✅ Track best profit
            if profit_amount > best_profit:
                best_profit = profit_amount
                best_profit_so_far = max(best_profit_so_far, profit_amount)
                best_profit_so_far_episode = episode + 1
                no_improvement_count = 0  # Reset counter when improving
            else:
                no_improvement_count += 1
            
            # ✅ Track history
            all_episodes.append(episode + 1)
            all_profits.append(profit_amount)
            
            # ✅ FIX OVERFITTING: Early stopping check mỗi 100 episodes
            if (episode + 1) % 100 == 0:
                # Check if performance is degrading significantly
                if len(all_profits) >= 1000:
                    recent_1000 = np.array(all_profits[-1000:])
                    prev_1000 = np.array(all_profits[-2000:-1000]) if len(all_profits) >= 2000 else None
                    
                    recent_avg = recent_1000.mean()
                    
                    if prev_1000 is not None:
                        prev_avg = prev_1000.mean()
                        degradation = ((recent_avg - prev_avg) / abs(prev_avg)) * 100 if prev_avg != 0 else 0
                        
                        if degradation < -20:  # Degraded by more than 20%
                            print(f"\n⚠️  WARNING: Performance degraded by {abs(degradation):.1f}% in last 1000 episodes")
                            print(f"   Previous avg: ${prev_avg:,.2f}, Recent avg: ${recent_avg:,.2f}")
                
                # Check if best profit was too early and current is much worse
                if best_profit_so_far > 50000:  # Only check if best is very high (possible overfitting)
                    if (episode + 1) - best_profit_so_far_episode > 1000:
                        if profit_amount < best_profit_so_far * 0.3:  # Current is < 30% of best
                            print(f"\n⚠️  WARNING: Best profit was at episode {best_profit_so_far_episode}")
                            print(f"   Current performance is much worse (${profit_amount:,.2f} vs ${best_profit_so_far:,.2f})")
            
            # ✅ Early stopping if no improvement for patience episodes (disabled nếu patience = 999999)
            if early_stopping_patience < 999999 and no_improvement_count >= early_stopping_patience and episode > start_episode + 500:
                print(f"\n{'='*70}")
                print(f"🛑 EARLY STOPPING")
                print(f"{'='*70}")
                print(f"   No improvement for {early_stopping_patience} episodes")
                print(f"   Best profit: ${best_profit_so_far:,.2f} (Episode {best_profit_so_far_episode})")
                print(f"   Current episode: {episode + 1}")
                print(f"   💡 Stopping training to prevent overfitting")
                print(f"{'='*70}\n")
                break
            
            # ✅ Timing
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)
            avg_episode_time = np.mean(episode_times[-100:])  # Average của 100 episodes gần nhất
            elapsed_time = time.time() - start_time
            remaining_episodes = n_episodes - (episode + 1)
            estimated_remaining = avg_episode_time * remaining_episodes
            
            # ✅ Progress logging - MỖI EPISODE để dễ theo dõi
            portfolio_value = mdp.get_portfolio_value()
            profit = ((portfolio_value - 10000.0) / 10000.0) * 100
            profit_amount = portfolio_value - 10000.0
            
            # ✅ Log mỗi episode (ngắn gọn, 1 dòng)
            q_info = ""
            if hasattr(agent, 'Q') and isinstance(agent.Q, dict):
                q_info = f"Q-states: {len(agent.Q):,} | "
            elif hasattr(agent, 'update_counter'):
                q_info = f"GPU Updates: {agent.update_counter:,} | "
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ep {episode + 1:5d}/{n_episodes} [{coin:3s}] | "
                  f"Reward: {episode_reward:8.2f} | Steps: {steps:5d} | "
                  f"Portfolio: ${portfolio_value:9.2f} | Profit: {profit:7.2f}% | "
                  f"{q_info}Epsilon: {agent.epsilon:.4f} | "
                  f"Time: {episode_time:.2f}s | ETA: {estimated_remaining/60:.1f}m", flush=True)
            
            # ✅ Early stop summary mỗi 10 episodes
            if (episode + 1) % 10 == 0:
                if early_stop_count > 0:
                    print(f"   ⚠️ Early stops (last 10 eps): {early_stop_count} episodes")
                    if early_stop_reasons:
                        reason_str = ", ".join([f"{k}: {v}" for k, v in early_stop_reasons.items()])
                        print(f"      Loss ranges: {reason_str}")
                    early_stop_count = 0  # Reset counter
                    early_stop_reasons = {}
            
            # Detailed log mỗi 100 episodes
            if (episode + 1) % 100 == 0:
                portfolio_value = mdp.get_portfolio_value()
                profit = ((portfolio_value - 10000.0) / 
                         10000.0 * 100)
                
                if hasattr(agent, 'Q') and isinstance(agent.Q, dict):
                    q_info = f"Q-states: {len(agent.Q):,}"
                elif hasattr(agent, 'update_counter'):
                    q_info = f"GPU Updates: {agent.update_counter:,}"
                else:
                    q_info = "DQN Network"
                
                print(f"\n{'='*70}", flush=True)
                print(f"✅ CHECKPOINT: Episode {episode + 1}/{n_episodes} [{coin}]", flush=True)
                print(f"  Reward: {episode_reward:.4f} | Steps: {steps:,}", flush=True)
                print(f"  Portfolio: ${portfolio_value:.2f} | Profit: {profit:.2f}%", flush=True)
                print(f"  Epsilon: {agent.epsilon:.4f} | {q_info}", flush=True)
                print(f"  Time: {elapsed_time/60:.1f} minutes elapsed", flush=True)
                print(f"  Avg speed: {avg_episode_time:.2f}s/episode", flush=True)
                if estimated_remaining > 0:
                    print(f"  Estimated completion: {estimated_remaining/60:.1f} minutes", flush=True)
                print(f"{'='*70}\n", flush=True)
            
            # ✅ Auto-save checkpoint mỗi 10 episodes và tự động xóa checkpoint cũ
            if (episode + 1) % episodes_per_save == 0:
                try:
                    # Tạo history dict với tracking đầy đủ
                    history = {
                        'episodes': all_episodes.copy(),  # ✅ Tất cả episodes đã train
                        'rewards': all_profits.copy(),    # ✅ Tất cả profits
                        'best_profit': best_profit_so_far,  # ✅ Best profit từ đầu đến giờ
                        'best_episode': best_profit_so_far_episode,
                        'epsilon': [agent.epsilon] * len(all_episodes),
                        'total_episodes': episode + 1      # ✅ Total episodes
                    }
                    
                    # Sử dụng auto_save_checkpoint để tự động cleanup checkpoints cũ
                    self.checkpoint_manager.auto_save_checkpoint(
                        agent,
                        episode + 1,
                        history,
                        mdp_state=None,
                        metadata={
                            'mode': 'sequential',
                            'coins': coin_names,
                            'episode_reward': episode_reward,
                            'portfolio_value': portfolio_value,
                            'profit': profit,
                            'profit_amount': profit_amount
                        },
                        save_interval=episodes_per_save,
                        keep_last_n=keep_last_n_checkpoints
                    )
                except Exception as e:
                    print(f"⚠️ Error saving checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    # ✅ Tiếp tục training dù checkpoint lỗi
                    continue
        
        print(f"\n✅ Training complete!")
        if hasattr(agent, 'Q') and isinstance(agent.Q, dict):
            print(f"   Final Q-table size: {len(agent.Q):,} states")
        elif hasattr(agent, 'update_counter'):
            print(f"   Total GPU Updates: {agent.update_counter:,}")
        total_time = time.time() - start_time
        print(f"   Total training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        
        return agent
    
    def _train_mixed(self, agent: QLearningAgent,
                    coins_data: dict,
                    n_episodes: int,
                    start_episode: int = 0,
                    checkpoint_data: dict = None) -> QLearningAgent:
        """
        Train mixed: kết hợp data của tất cả coins, shuffle random
        
        Args:
            agent: Q-Learning agent
            coins_data: Dict of coin dataframes
            n_episodes: Total episodes to train
            start_episode: Episode to start from (for resume)
            checkpoint_data: Checkpoint data for resume
        """
        print("\n🔀 Mixed Training Mode")
        print("   Training on shuffled multi-coin data")
        
        # ✅ Lấy danh sách coins
        coin_names = list(coins_data.keys())
        
        # Combine all data
        combined = pd.concat(coins_data.values(), ignore_index=False)
        
        # ✅ Sort by index (timestamp) để giữ thứ tự thời gian
        combined = combined.sort_index()
        
        # ✅ Remove duplicates nếu có
        combined = combined[~combined.index.duplicated(keep='last')]
        
        print(f"   Total samples: {len(combined):,}")
        print(f"   Date range: {combined.index.min()} to {combined.index.max()}")
        
        # ✅ MAJOR FIX: Create MDP với Risk Management DISABLED
        from src.config.config import RISK_MANAGEMENT_PARAMS
        mdp = TradingMDP(
            combined,
            initial_balance=10000.0,
            transaction_cost=0.001,
            hold_penalty=0.0001,
            interval='1h',
            stop_loss_pct=0.10,
            max_position_pct=RISK_MANAGEMENT_PARAMS['max_position_pct'],
            max_loss_pct=RISK_MANAGEMENT_PARAMS['max_loss_pct'],
            trailing_stop_pct=RISK_MANAGEMENT_PARAMS['trailing_stop_pct'],
            enable_risk_management=False  # ✅ MAJOR FIX: Disable early stopping
        )
        
        episodes_per_save = 10  # ✅ Lưu mỗi 10 episodes
        keep_last_n_checkpoints = 10  # ✅ Giữ lại 10 checkpoints gần nhất (100 episodes)
        
        # ✅ Timing tracking
        start_time = time.time()
        episode_times = []
        
        # ✅ Track best profit và total episodes cho checkpoint - Resume từ checkpoint nếu có
        if checkpoint_data and checkpoint_data.get('history'):
            history = checkpoint_data['history']
            best_profit = history.get('best_profit', float('-inf'))
            all_episodes = history.get('episodes', [])
            all_profits = history.get('rewards', [])  # rewards là profits
            best_episode_num = history.get('best_episode', 0)
            print(f"   📊 Resumed tracking: {len(all_episodes)} episodes, best profit: ${best_profit:.2f}")
        else:
            best_profit = float('-inf')
            best_episode_num = 0
            all_episodes = []
            all_profits = []
        
        # ✅ DISABLE EARLY STOPPING: Set patience rất cao để model chạy hết episodes
        early_stopping_patience = 999999  # ✅ Disable early stopping - model sẽ chạy hết 5000 episodes
        best_profit_so_far = best_profit
        best_profit_so_far_episode = best_episode_num
        no_improvement_count = 0
        
        for episode in range(start_episode, n_episodes):
            episode_start = time.time()
            
            try:
                state = mdp.reset()
                done = False
                episode_reward = 0
                steps = 0
                
                max_steps = len(combined)
                last_info = None  # ✅ Track info từ step cuối cùng
                while not done and steps < max_steps:
                    action = agent.select_action(state)
                    next_state, reward, done, step_info = mdp.step(action)
                    last_info = step_info  # ✅ Lưu info từ step cuối cùng
                    agent.update_q_value(state, action, reward, next_state, done)
                    
                    # ✅ V4: Force mini-batch update mỗi 20 steps để GPU không idle
                    # (Đặc biệt quan trọng khi episodes ngắn)
                    if hasattr(agent, 'batch_buffer') and steps % 20 == 0 and len(agent.batch_buffer) > 0:
                        if hasattr(agent, 'min_batch_size'):
                            if len(agent.batch_buffer) >= agent.min_batch_size:
                                agent._update_batch()
                                agent.update_counter += 1
                    
                    state = next_state
                    episode_reward += reward
                    steps += 1
                    
            except Exception as e:
                print(f"⚠️ Error in episode {episode + 1}: {e}")
                continue
            
            # ✅ Flush batch buffer nếu dùng DQN (đảm bảo tất cả samples được update)
            if hasattr(agent, 'flush_batch'):
                agent.flush_batch()
            
            # ✅ V4: Track early stops để summary (không spam log)
            if last_info and last_info.get('episode_ended_early', False):
                early_stop_reason = last_info.get('early_stop_reason')
                if early_stop_reason:
                    early_stop_count += 1
                    # Extract loss percentage từ reason
                    if 'Max loss exceeded:' in early_stop_reason:
                        try:
                            loss_pct = float(early_stop_reason.split(':')[1].strip().replace('%', ''))
                            loss_range = f"{int(loss_pct//10)*10}-{int(loss_pct//10)*10+10}%"
                            early_stop_reasons[loss_range] = early_stop_reasons.get(loss_range, 0) + 1
                        except:
                            pass
            
            agent.decay_epsilon()
            
            # ✅ Tính portfolio value và profit
            portfolio_value = mdp.get_portfolio_value()
            profit = ((portfolio_value - 10000.0) / 10000.0) * 100
            profit_amount = portfolio_value - 10000.0  # Profit in USD
            
            # ✅ Track best profit
            if profit_amount > best_profit:
                best_profit = profit_amount
                best_profit_so_far = max(best_profit_so_far, profit_amount)
                best_profit_so_far_episode = episode + 1
                no_improvement_count = 0  # Reset counter when improving
            else:
                no_improvement_count += 1
            
            # ✅ Track history
            all_episodes.append(episode + 1)
            all_profits.append(profit_amount)
            
            # ✅ V5: Early stopping check mỗi 100 episodes
            if (episode + 1) % 100 == 0:
                # Check if performance is degrading significantly
                if len(all_profits) >= 1000:
                    recent_1000 = np.array(all_profits[-1000:])
                    prev_1000 = np.array(all_profits[-2000:-1000]) if len(all_profits) >= 2000 else None
                    
                    recent_avg = recent_1000.mean()
                    
                    if prev_1000 is not None:
                        prev_avg = prev_1000.mean()
                        degradation = ((recent_avg - prev_avg) / abs(prev_avg)) * 100 if prev_avg != 0 else 0
                        
                        if degradation < -20:  # Degraded by more than 20%
                            print(f"\n⚠️  WARNING: Performance degraded by {abs(degradation):.1f}% in last 1000 episodes")
                            print(f"   Previous avg: ${prev_avg:,.2f}, Recent avg: ${recent_avg:,.2f}")
                
                # Check if best profit was too early and current is much worse
                if best_profit_so_far > 50000:  # Only check if best is very high (possible overfitting)
                    if (episode + 1) - best_profit_so_far_episode > 1000:
                        if profit_amount < best_profit_so_far * 0.3:  # Current is < 30% of best
                            print(f"\n⚠️  WARNING: Best profit was at episode {best_profit_so_far_episode}")
                            print(f"   Current performance is much worse (${profit_amount:,.2f} vs ${best_profit_so_far:,.2f})")
            
            # ✅ Early stopping if no improvement for patience episodes (disabled nếu patience = 999999)
            if early_stopping_patience < 999999 and no_improvement_count >= early_stopping_patience and episode > start_episode + 500:
                print(f"\n{'='*70}")
                print(f"🛑 EARLY STOPPING")
                print(f"{'='*70}")
                print(f"   No improvement for {early_stopping_patience} episodes")
                print(f"   Best profit: ${best_profit_so_far:,.2f} (Episode {best_profit_so_far_episode})")
                print(f"   Current episode: {episode + 1}")
                print(f"   💡 Stopping training to prevent overfitting")
                print(f"{'='*70}\n")
                break
            
            # ✅ Timing
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)
            avg_episode_time = np.mean(episode_times[-100:])
            elapsed_time = time.time() - start_time
            remaining_episodes = n_episodes - (episode + 1)
            estimated_remaining = avg_episode_time * remaining_episodes
            
            # ✅ Progress logging - MỖI EPISODE để dễ theo dõi
            portfolio_value = mdp.get_portfolio_value()
            profit = ((portfolio_value - 10000.0) / 10000.0) * 100
            profit_amount = portfolio_value - 10000.0
            
            # ✅ Log mỗi episode (ngắn gọn, 1 dòng)
            q_info = ""
            if hasattr(agent, 'Q') and isinstance(agent.Q, dict):
                q_info = f"Q-states: {len(agent.Q):,} | "
            elif hasattr(agent, 'update_counter'):
                q_info = f"GPU Updates: {agent.update_counter:,} | "
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ep {episode + 1:5d}/{n_episodes} [Mixed] | "
                  f"Reward: {episode_reward:8.2f} | Steps: {steps:5d} | "
                  f"Portfolio: ${portfolio_value:9.2f} | Profit: {profit:7.2f}% | "
                  f"{q_info}Epsilon: {agent.epsilon:.4f} | "
                  f"Time: {episode_time:.2f}s | ETA: {estimated_remaining/60:.1f}m", flush=True)
            
            # ✅ Early stop summary mỗi 10 episodes
            if (episode + 1) % 10 == 0:
                if early_stop_count > 0:
                    print(f"   ⚠️ Early stops (last 10 eps): {early_stop_count} episodes")
                    if early_stop_reasons:
                        reason_str = ", ".join([f"{k}: {v}" for k, v in early_stop_reasons.items()])
                        print(f"      Loss ranges: {reason_str}")
                    early_stop_count = 0  # Reset counter
                    early_stop_reasons = {}
            
            # Detailed log mỗi 100 episodes
            if (episode + 1) % 100 == 0:
                portfolio_value = mdp.get_portfolio_value()
                profit = ((portfolio_value - 10000.0) / 
                         10000.0 * 100)
                
                if hasattr(agent, 'Q') and isinstance(agent.Q, dict):
                    q_info = f"Q-states: {len(agent.Q):,}"
                elif hasattr(agent, 'update_counter'):
                    q_info = f"GPU Updates: {agent.update_counter:,}"
                else:
                    q_info = "DQN Network"
                
                print(f"\n{'='*70}", flush=True)
                print(f"✅ CHECKPOINT: Episode {episode + 1}/{n_episodes} [Mixed]", flush=True)
                print(f"  Reward: {episode_reward:.4f} | Steps: {steps:,}", flush=True)
                print(f"  Portfolio: ${portfolio_value:.2f} | Profit: {profit:.2f}%", flush=True)
                print(f"  Epsilon: {agent.epsilon:.4f} | {q_info}", flush=True)
                print(f"  Time: {elapsed_time/60:.1f} minutes elapsed", flush=True)
                print(f"  Avg speed: {avg_episode_time:.2f}s/episode", flush=True)
                if estimated_remaining > 0:
                    print(f"  Estimated completion: {estimated_remaining/60:.1f} minutes", flush=True)
                print(f"{'='*70}\n", flush=True)
            
            # ✅ Auto-save checkpoint mỗi 10 episodes và tự động xóa checkpoint cũ
            if (episode + 1) % episodes_per_save == 0:
                try:
                    # Tạo history dict với tracking đầy đủ
                    history = {
                        'episodes': all_episodes.copy(),  # ✅ Tất cả episodes đã train
                        'rewards': all_profits.copy(),    # ✅ Tất cả profits
                        'best_profit': best_profit_so_far,  # ✅ Best profit từ đầu đến giờ
                        'best_episode': best_profit_so_far_episode,
                        'epsilon': [agent.epsilon] * len(all_episodes),
                        'total_episodes': episode + 1      # ✅ Total episodes
                    }
                    
                    # Sử dụng auto_save_checkpoint để tự động cleanup checkpoints cũ
                    self.checkpoint_manager.auto_save_checkpoint(
                        agent,
                        episode + 1,
                        history,
                        mdp_state=None,
                        metadata={
                            'mode': 'mixed',
                            'coins': coin_names,
                            'episode_reward': episode_reward,
                            'portfolio_value': portfolio_value,
                            'profit': profit,
                            'profit_amount': profit_amount
                        },
                        save_interval=episodes_per_save,
                        keep_last_n=keep_last_n_checkpoints
                    )
                except Exception as e:
                    print(f"⚠️ Error saving checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    # ✅ Tiếp tục training dù checkpoint lỗi
                    continue
        
        print(f"\n✅ Training complete!")
        if hasattr(agent, 'Q') and isinstance(agent.Q, dict):
            print(f"   Final Q-table size: {len(agent.Q):,} states")
        elif hasattr(agent, 'update_counter'):
            print(f"   Total GPU Updates: {agent.update_counter:,}")
        total_time = time.time() - start_time
        print(f"   Total training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        
        return agent


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Q-Learning on multi-coin data')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to multi-coin CSV file')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of training episodes (default: 5000)')
    parser.add_argument('--mode', type=str, default='sequential',
                       choices=['sequential', 'mixed'],
                       help='Training mode (default: sequential)')
    parser.add_argument('--use-dqn', action='store_true',
                       help='Use Deep Q-Network (DQN) with GPU instead of Tabular Q-Learning')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from checkpoint (start fresh)')
    parser.add_argument('--train-ratio', type=float, default=None,
                       help='DEPRECATED: Ratio of data for training. Use --train-start/--train-end instead.')
    parser.add_argument('--train-start', type=str, default='2020-01-01',
                       help='Training start date (default: 2020-01-01)')
    parser.add_argument('--train-end', type=str, default='2023-12-31',
                       help='Training end date (default: 2023-12-31)')
    parser.add_argument('--val-start', type=str, default='2024-01-01',
                       help='Validation start date (default: 2024-01-01)')
    parser.add_argument('--val-end', type=str, default='2024-06-30',
                       help='Validation end date (default: 2024-06-30)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MultiCoinTrainer()
    
    # Load data - ✅ TIME-SERIES STRATEGY: Load tất cả 5 coins từ multi_coin_1h.csv
    df = trainer.load_multi_coin_data(args.data)
    
    if df is None:
        print("\n❌ Cannot load data. Please prepare data first:")
        print("   python src/data/prepare_multi_coin_data.py")
        return
    
    # Train
    agent = trainer.train_multi_coin(
        df,
        n_episodes=args.episodes,
        mode=args.mode,
        use_dqn=args.use_dqn,  # ✅ GPU support
        resume=not args.no_resume,  # ✅ Resume từ checkpoint (default: True)
        train_start=args.train_start,  # ✅ TIME-SERIES SPLIT: Training period
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        train_ratio=args.train_ratio  # Fallback to ratio if provided
    )
    
    print("\n✅ Training complete!")
    print(f"   Checkpoints saved to: {trainer.checkpoint_manager.checkpoint_dir}")


if __name__ == "__main__":
    main()
