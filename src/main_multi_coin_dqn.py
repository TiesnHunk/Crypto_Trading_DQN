"""
🚀 DQN TRAINING SCRIPT - Multi-Coin Deep Q-Network
Replaces Tabular Q-Learning with DQN to solve State Collapse issue

Changes from main_multi_coin.py:
- ✅ Replace QLearningAgent → DQNAgent
- ✅ Keep TradingMDP (paper's trend-based rewards)
- ✅ Add Experience Replay training loop
- ✅ Add Target Network updates
- ✅ Track training loss
- ✅ GPU acceleration with CUDA
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

# ✅ FIX OUTPUT BUFFERING: Force unbuffered output
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    import os
    os.environ['PYTHONUNBUFFERED'] = '1'

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.mdp_trading import TradingMDP
from src.models.dqn_agent import DQNAgent  # ✅ NEW: Import DQN Agent
from src.utils.indicators import TechnicalIndicators
from src.config import config
from src.config.config import DEVICE, USE_GPU


class DQNMultiCoinTrainer:
    """
    🚀 DQN Trainer cho multi-coin trading
    
    Features:
    - Deep Q-Network với continuous states (no binning)
    - Experience Replay Buffer (100K capacity)
    - Target Network (updated every 1000 steps)
    - Epsilon-greedy exploration
    - GPU acceleration
    """
    
    def __init__(self, checkpoint_dir: str = None):
        """
        Initialize DQN trainer
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent / 'checkpoints_dqn'
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n💾 Checkpoint directory: {self.checkpoint_dir}")
        
    def load_multi_coin_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load multi-coin data
        
        Args:
            filepath: Path to multi-coin CSV file
            
        Returns:
            DataFrame với tất cả coins
        """
        if filepath is None:
            filepath = Path(__file__).parent.parent / 'data' / 'raw' / 'multi_coin_1h.csv'
            if not filepath.exists():
                filepath = Path('data') / 'raw' / 'multi_coin_1h.csv'
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            print("   Please run: python src/data/prepare_multi_coin_data.py")
            return None
        
        print(f"\n📥 Loading multi-coin data from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        print(f"✅ Loaded {len(df):,} rows")
        print(f"   Coins: {df['coin'].unique()}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        
        # Add technical indicators if missing
        required_cols = ['rsi', 'macd_hist', 'trend', 'bb_upper', 'bb_lower', 'volatility']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\n📊 Adding technical indicators (missing: {missing_cols})...")
            df = TechnicalIndicators.add_all_indicators(df)
            
            # Fill NaN values
            df['rsi'] = df['rsi'].fillna(50.0)
            df['macd_hist'] = df['macd_hist'].fillna(0.0)
            df['volatility'] = df['volatility'].fillna(0.0)
            
            df = df.dropna()
            print(f"✅ Indicators added. Remaining rows: {len(df):,}")
        
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
            
            # Remove 'coin' column
            if 'coin' in coin_df.columns:
                coin_df = coin_df.drop('coin', axis=1)
            
            # Ensure timestamp index
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
                     val_end: str = '2024-06-30') -> tuple:
        """
        Time-series split: Training và Validation data
        
        Args:
            df: Multi-coin DataFrame
            train_start: Training start date
            train_end: Training end date
            val_start: Validation start date
            val_end: Validation end date
            
        Returns:
            (train_df, validation_df)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have timestamp index")
        
        df = df.sort_index()
        
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        val_start_dt = pd.to_datetime(val_start)
        val_end_dt = pd.to_datetime(val_end)
        
        train_df = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)].copy()
        validation_df = df[(df.index >= val_start_dt) & (df.index <= val_end_dt)].copy()
        
        train_coins = train_df['coin'].value_counts() if 'coin' in train_df.columns else {}
        val_coins = validation_df['coin'].value_counts() if 'coin' in validation_df.columns else {}
        
        print(f"\n{'='*70}")
        print(f"📅 TIME-SERIES SPLIT (DQN Training)")
        print(f"{'='*70}")
        print(f"   Training Period:   {train_df.index.min()} to {train_df.index.max()}")
        print(f"   Training Samples:  {len(train_df):,}")
        if len(train_coins) > 0:
            print(f"   Training Coins:    {dict(train_coins)}")
        print(f"   Validation Period: {validation_df.index.min()} to {validation_df.index.max()}")
        print(f"   Validation Samples: {len(validation_df):,}")
        if len(val_coins) > 0:
            print(f"   Validation Coins:  {dict(val_coins)}")
        print(f"{'='*70}\n")
        
        return train_df, validation_df
    
    def train_dqn(self, df: pd.DataFrame, 
                  n_episodes: int = 5000,
                  train_start: str = '2020-01-01',
                  train_end: str = '2023-12-31',
                  val_start: str = '2024-01-01',
                  val_end: str = '2024-06-30',
                  resume: bool = True) -> DQNAgent:
        """
        🚀 Train DQN agent on multi-coin data
        
        Args:
            df: Multi-coin DataFrame
            n_episodes: Number of training episodes
            train_start: Training start date
            train_end: Training end date
            val_start: Validation start date
            val_end: Validation end date
            resume: Resume from checkpoint if available
            
        Returns:
            Trained DQN agent
        """
        print("\n" + "="*70)
        print(f"🚀 DQN MULTI-COIN TRAINING (Deep Q-Network)")
        print(f"   Episodes: {n_episodes}")
        print(f"   Architecture: DQN with Experience Replay")
        print(f"   Strategy: All 5 coins, time-series split")
        print("="*70)
        
        # GPU Info
        print(f"\n💻 GPU Status:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Device: {DEVICE}")
        
        # Time-series split
        train_df, validation_df = self.split_by_time(
            df, 
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end
        )
        
        # Save validation data for later
        self.validation_df = validation_df
        
        # Split training data by coin
        coins_data = self.split_by_coin(train_df)
        print(f"\n📊 Training on {len(coins_data)} coins:")
        for coin, coin_df in coins_data.items():
            print(f"   {coin}: {len(coin_df):,} samples")
        
        # ✅ Initialize or load DQN agent
        start_episode = 0
        agent = None
        
        if resume:
            checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pkl'
            if checkpoint_path.exists():
                print(f"\n🔄 Attempting to load checkpoint from {checkpoint_path}...")
                agent = DQNAgent(
                    state_dim=12,  # 🆕 V7: Enhanced state space (8→12 with market regime detection)
                    action_dim=3,  # Buy, Sell, Hold
                    device=DEVICE
                )
                try:
                    checkpoint_data = agent.load_checkpoint(str(checkpoint_path))
                    if checkpoint_data:
                        start_episode = checkpoint_data['episode']
                        print(f"✅ Resumed from episode {start_episode}")
                        print(f"   Epsilon: {agent.epsilon:.4f}")
                        print(f"   Replay buffer: {len(agent.replay_buffer)} transitions")
                except (RuntimeError, KeyError) as e:
                    print(f"⚠️ Checkpoint incompatible with new architecture: {e}")
                    print(f"   Old checkpoint has different state_dim (likely 8)")
                    print(f"   New model has state_dim=12 (market regime detection)")
                    print(f"   Starting fresh training from episode 0...")
                    agent = None  # Force new agent creation
        
        if agent is None:
            print(f"\n🆕 Creating new DQN agent with ENHANCED configuration...")
            agent = DQNAgent(
                state_dim=12,  # 🆕 V7: Enhanced state space (8→12 with market regime detection)
                action_dim=3,  # Buy, Sell, Hold
                learning_rate=0.001,
                gamma=0.95,  # 🆕 V7: Reduced from 0.99 to 0.95 (less long-term bias)
                epsilon_start=1.0,
                epsilon_end=0.05,  # 🆕 V7: Increased from 0.01 to 0.05 (more exploration)
                epsilon_decay=0.995,
                buffer_capacity=100000,
                batch_size=64,
                target_update_frequency=1000,
                device=DEVICE
            )
            print(f"✅ DQN Agent created with V7 enhancements")
            print(f"   State dimensions: 12 (added market regime detection)")
            print(f"   Gamma: 0.95 (market regime adaptation)")
            print(f"   Epsilon end: 0.05 (better exploration)")
            print(f"   Network parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
            print(f"   Device: {agent.device}")
        
        # ✅ Train DQN
        agent = self._train_sequential_dqn(
            agent, 
            coins_data, 
            n_episodes, 
            start_episode
        )
        
        return agent
    
    def _train_sequential_dqn(self, agent: DQNAgent,
                             coins_data: dict,
                             n_episodes: int,
                             start_episode: int = 0) -> DQNAgent:
        """
        🎯 Train DQN sequentially: each episode trains on 1 random coin
        
        DQN Training Loop:
        1. Select action using epsilon-greedy
        2. Execute action in environment (MDP)
        3. Store transition in replay buffer
        4. Sample random batch from buffer
        5. Train policy network with MSE loss
        6. Update target network every N steps
        
        Args:
            agent: DQN agent
            coins_data: Dict of coin dataframes
            n_episodes: Total episodes
            start_episode: Starting episode (for resume)
            
        Returns:
            Trained DQN agent
        """
        print("\n🔄 DQN Sequential Training Mode")
        print("   Each episode trains on 1 random coin")
        print("   Experience replay: Store and sample from buffer")
        print("   Target network: Updated every 1000 steps")
        print("   💾 Auto-save: Every episode (for safe resume)")
        
        coin_names = list(coins_data.keys())
        save_interval = 1  # Save checkpoint EVERY episode (was 100)
        
        # Tracking
        start_time = time.time()
        episode_times = []
        best_profit = float('-inf')
        best_episode = 0
        
        # History tracking
        all_episodes = []
        all_profits = []
        all_losses = []
        all_mdds = []  # 🆕 V5: Track MDD for each episode
        all_annualized_returns = []  # 🆕 V5: Track Annualized Return for each episode
        
        print(f"\n🎬 Starting training from episode {start_episode + 1}...\n")
        
        for episode in range(start_episode, n_episodes):
            episode_start = time.time()
            
            # Random select coin
            coin = np.random.choice(coin_names)
            coin_df = coins_data[coin]
            
            # Create MDP environment
            mdp = TradingMDP(
                coin_df,
                initial_balance=10000.0,
                transaction_cost=0.0001,  # Realistic 0.01% fee
                hold_penalty=0.00001,  # 🆕 V7: Reduced from 0.001 to 0.00001 (less conservative)
                interval='1h',
                stop_loss_pct=0.15,  # Increased from 0.10 (10%) to 0.15 (15%)
                enable_risk_management=False  # Paper uses 100% balance per trade
            )
            
            # Train 1 episode
            try:
                state = mdp.reset()
                
                # 🆕 V5.1: Update epsilon in MDP for epsilon-aware emergency stop
                mdp.set_epsilon(agent.epsilon)
                
                done = False
                episode_reward = 0
                episode_losses = []
                steps = 0
                max_steps = len(coin_df)
                
                while not done and steps < max_steps:
                    # ✅ DQN: Select action with epsilon-greedy
                    action = agent.select_action(state, agent.epsilon)
                    
                    # Execute action
                    next_state, reward, done, info = mdp.step(action)
                    
                    # ✅ DQN: Store transition in replay buffer
                    agent.store_transition(state, action, reward, next_state, done)
                    
                    # ✅ DQN: Train if buffer has enough samples
                    if len(agent.replay_buffer) >= agent.batch_size:
                        loss = agent.train_step()
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    state = next_state
                    episode_reward += reward
                    steps += 1
                
            except Exception as e:
                print(f"⚠️ Error in episode {episode + 1} with {coin}: {e}")
                continue
            
            # ✅ DQN: Decay epsilon
            agent.update_epsilon()
            
            # Calculate profit
            portfolio_value = mdp.get_portfolio_value()
            profit = ((portfolio_value - 10000.0) / 10000.0) * 100
            profit_amount = portfolio_value - 10000.0
            
            # Track best profit
            if profit_amount > best_profit:
                best_profit = profit_amount
                best_episode = episode + 1
                # Save best model
                best_path = self.checkpoint_dir / 'checkpoint_best.pkl'
                elapsed_time_now = time.time() - start_time
                agent.save_checkpoint(
                    str(best_path),
                    episode + 1,
                    {
                        'best_profit': best_profit, 
                        'best_episode': best_episode,
                        'training_time': elapsed_time_now
                    }
                )
            
            # Track history
            all_episodes.append(episode + 1)
            all_profits.append(profit_amount)
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            all_losses.append(avg_loss)
            
            # 🆕 V5: Extract MDD & Annualized Return from final info
            max_mdd = info.get('max_mdd', 0.0)
            annualized_return = info.get('annualized_return', 0.0)
            all_mdds.append(max_mdd)  # Track for statistics
            all_annualized_returns.append(annualized_return)  # Track for statistics
            
            # Timing
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)
            avg_episode_time = np.mean(episode_times[-100:])
            elapsed_time = time.time() - start_time
            remaining_episodes = n_episodes - (episode + 1)
            estimated_remaining = avg_episode_time * remaining_episodes
            
            # ✅ Log mỗi episode (1 dòng) - UPDATED WITH MDD & ANNUALIZED RETURN
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ep {episode + 1:5d}/{n_episodes} [{coin:3s}] | "
                  f"Reward: {episode_reward:8.2f} | Steps: {steps:5d} | "
                  f"Portfolio: ${portfolio_value:9.2f} | Profit: {profit:7.2f}% | "
                  f"MDD: {max_mdd:6.2f}% | Ann.Ret: {annualized_return:7.2f}% | "
                  f"Loss: {avg_loss:7.4f} | Buffer: {len(agent.replay_buffer):6d} | "
                  f"Epsilon: {agent.epsilon:.4f} | Time: {episode_time:.2f}s | ETA: {estimated_remaining/60:.1f}m",
                  flush=True)
            
            # Detailed log every 100 episodes
            if (episode + 1) % 100 == 0:
                recent_100_profits = all_profits[-100:] if len(all_profits) >= 100 else all_profits
                avg_profit_100 = np.mean(recent_100_profits)
                recent_100_losses = all_losses[-100:] if len(all_losses) >= 100 else all_losses
                avg_loss_100 = np.mean(recent_100_losses)
                
                # 🆕 V5: MDD & Annualized Return statistics
                recent_100_mdds = all_mdds[-100:] if len(all_mdds) >= 100 else all_mdds
                avg_mdd_100 = np.mean(recent_100_mdds) if recent_100_mdds else 0.0
                max_mdd_100 = np.max(recent_100_mdds) if recent_100_mdds else 0.0
                recent_100_ann_ret = all_annualized_returns[-100:] if len(all_annualized_returns) >= 100 else all_annualized_returns
                avg_ann_ret_100 = np.mean(recent_100_ann_ret) if recent_100_ann_ret else 0.0
                
                print(f"\n{'='*70}", flush=True)
                print(f"✅ CHECKPOINT: Episode {episode + 1}/{n_episodes}", flush=True)
                print(f"   Last episode: [{coin}] Reward: {episode_reward:.2f} | Steps: {steps:,}", flush=True)
                print(f"   Portfolio: ${portfolio_value:.2f} | Profit: {profit:.2f}%", flush=True)
                print(f"   Avg profit (last 100): ${avg_profit_100:.2f} ({(avg_profit_100/10000)*100:.2f}%)", flush=True)
                print(f"   Avg loss (last 100): {avg_loss_100:.4f}", flush=True)
                print(f"   🆕 Avg MDD (last 100): {avg_mdd_100:.2f}% | Max MDD: {max_mdd_100:.2f}%", flush=True)
                print(f"   🆕 Avg Ann.Return (last 100): {avg_ann_ret_100:.2f}%", flush=True)
                print(f"   Best profit: ${best_profit:.2f} at episode {best_episode}", flush=True)
                print(f"   Epsilon: {agent.epsilon:.4f}", flush=True)
                print(f"   Replay buffer: {len(agent.replay_buffer):,}/{agent.replay_buffer.capacity:,} ({len(agent.replay_buffer)/agent.replay_buffer.capacity*100:.1f}%)", flush=True)
                print(f"   Time: {elapsed_time/60:.1f} min elapsed | {avg_episode_time:.2f}s/episode", flush=True)
                if estimated_remaining > 0:
                    print(f"   ETA: {estimated_remaining/60:.1f} minutes (~{estimated_remaining/3600:.1f} hours)", flush=True)
                print(f"{'='*70}\n", flush=True)
            
            # Save checkpoint every save_interval episodes (now every episode for safety)
            if (episode + 1) % save_interval == 0:
                checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pkl'
                metadata = {
                    'all_episodes': all_episodes,
                    'all_profits': all_profits,
                    'all_losses': all_losses,
                    'best_profit': best_profit,
                    'best_episode': best_episode,
                    'training_time': elapsed_time
                }
                agent.save_checkpoint(str(checkpoint_path), episode + 1, metadata)
                # Don't print every episode to avoid spam (only print every 10 episodes)
                if (episode + 1) % 10 == 0:
                    print(f"   💾 Checkpoint auto-saved: Episode {episode + 1}", flush=True)
        
        # Final summary
        final_portfolio = mdp.get_portfolio_value() if 'mdp' in locals() else 10000.0
        print(f"\n{'='*70}")
        print(f"🏁 TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"   Total episodes: {n_episodes}")
        print(f"   Total time: {(time.time() - start_time)/3600:.2f} hours")
        print(f"   Best profit: ${best_profit:.2f} at episode {best_episode}")
        print(f"   Final epsilon: {agent.epsilon:.4f}")
        print(f"   Replay buffer: {len(agent.replay_buffer):,} transitions")
        print(f"   Final portfolio: ${final_portfolio:.2f}")
        print(f"{'='*70}\n")
        
        # Save final checkpoint
        final_path = self.checkpoint_dir / 'checkpoint_final.pkl'
        metadata = {
            'all_episodes': all_episodes,
            'all_profits': all_profits,
            'all_losses': all_losses,
            'best_profit': best_profit,
            'best_episode': best_episode,
            'training_time': time.time() - start_time,
            'training_complete': True
        }
        agent.save_checkpoint(str(final_path), n_episodes, metadata)
        print(f"💾 Final checkpoint saved: {final_path}\n")
        
        return agent


def main():
    """
    🚀 Main training function
    """
    print("\n" + "="*70)
    print("🚀 DQN MULTI-COIN TRAINING SCRIPT")
    print("="*70)
    print("   Replace Tabular Q-Learning → Deep Q-Network")
    print("   Solve State Collapse with continuous states")
    print("   Keep paper's trend-based rewards (MDP)")
    print("="*70)
    
    # Initialize trainer
    trainer = DQNMultiCoinTrainer()
    
    # Load data
    df = trainer.load_multi_coin_data()
    if df is None:
        print("❌ Failed to load data. Exiting...")
        return
    
    # Train DQN
    agent = trainer.train_dqn(
        df,
        n_episodes=5000,  # Start with 5000 episodes
        train_start='2020-01-01',
        train_end='2023-12-31',
        val_start='2024-01-01',
        val_end='2024-06-30',
        resume=True  # Resume from checkpoint if available
    )
    
    print("\n✅ Training complete!")
    print(f"   Use validate_time_series.py to test on validation data")
    print(f"   Checkpoint directory: {trainer.checkpoint_dir}")
    print(f"   Best model saved: checkpoint_best.pkl")
    print(f"   Latest model saved: checkpoint_latest.pkl\n")


if __name__ == '__main__':
    main()
