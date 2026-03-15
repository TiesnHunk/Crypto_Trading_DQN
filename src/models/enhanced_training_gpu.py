"""
Enhanced Training Module with GPU Support
Module huấn luyện cải tiến với hỗ trợ GPU và thời gian training dài hơn
"""

import numpy as np
from collections import defaultdict
import pandas as pd
 
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.config import (
    TRAINING_EPISODES, 
    EARLY_STOPPING_PATIENCE, 
    LOG_INTERVAL,
    DEVICE,
    SAVE_PLOTS,
    PLOT_DIR
)
from src.utils.checkpoint import TrainingCheckpoint


def train_with_gpu_acceleration(Q, mdp, n_episodes=None, patience=None, 
                                min_improvement=0.001, verbose=True,
                                checkpoint_dir="checkpoints",
                                checkpoint_interval=100,
                                resume_from_checkpoint=True,
                                early_stopping: bool = True):
    """
    Train với GPU acceleration và early stopping
    Hỗ trợ cả Q-table và Deep Q-Network
    
    Args:
        Q: Q-table (dict) hoặc QLearningAgent với neural network
        mdp: MDP environment
        n_episodes: Số episodes tối đa (default từ config)
        patience: Số episodes không cải thiện (default từ config)
        min_improvement: Độ cải thiện tối thiểu
        verbose: In thông tin training
        checkpoint_dir: Thư mục lưu checkpoint
        checkpoint_interval: Lưu checkpoint mỗi N episodes
    resume_from_checkpoint: Có tiếp tục từ checkpoint không
    early_stopping: Bật/tắt dừng sớm khi không cải thiện. Nếu False, sẽ chỉ dừng khi đạt tối đa n_episodes.
    
    Returns:
        Training history
    """
    if n_episodes is None:
        n_episodes = TRAINING_EPISODES
    if patience is None:
        patience = EARLY_STOPPING_PATIENCE
    
    # state_dim = len(mdp.data)  # unused
    n_actions = 3
    
    # Kiểm tra Q là dict (tabular) hay QLearningAgent (DQN)
    is_dqn = hasattr(Q, 'select_action') and hasattr(Q, 'update_q_value')
    
    # Set gamma cho cả 2 trường hợp
    if is_dqn:
        agent = Q
        gamma = agent.gamma  # Lấy gamma từ agent
        if verbose:
            print("\n🧠 Using Deep Q-Network (DQN) with GPU")
            print(f"   Device: {agent.device}")
    else:
        if verbose:
            print("\n📊 Using Tabular Q-Learning (CPU only)")
        
        alpha = 0.1
        gamma = 0.95
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.01
    
    history = {
        'episodes': [],
        'profits': [],
        'epsilon': [],
        'returns': [],
        'converged': False,
        'best_episode': 0,
        'best_profit': -np.inf
    }
    
    best_profit = -np.inf
    patience_counter = 0
    start_episode = 0
    
    # Khởi tạo checkpoint manager
        
    checkpoint_manager = TrainingCheckpoint(checkpoint_dir)
    
    # Load checkpoint nếu có
    if resume_from_checkpoint:
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            loaded_Q = checkpoint_data['Q']
            # Convert back to defaultdict if it's tabular Q-learning
            if not is_dqn:
                Q = defaultdict(lambda: np.zeros(n_actions))
                Q.update(loaded_Q)
            else:
                Q = loaded_Q
            
            start_episode = checkpoint_data['episode'] + 1
            history = checkpoint_data['history']
            best_profit = history.get('best_profit', -np.inf)
            patience_counter = checkpoint_data.get('patience_counter', 0)
        
            if verbose:
                print("\n🔄 Resuming from checkpoint")
                print(f"   Starting from episode: {start_episode}")
                print(f"   Best profit so far: {best_profit:.4f}")
                if not is_dqn:
                    print(f"   Loaded {len(loaded_Q)} states into Q-table")
    
    if verbose:
        print("\n🏋️ Training with GPU Acceleration")
        print("="*80)
        print(f"   Device: {DEVICE}")
        print(f"   Total Episodes: {n_episodes}")
        print(f"   Starting Episode: {start_episode}")
        if early_stopping:
            print(f"   Early Stopping Patience: {patience}")
        else:
            print("   Early Stopping: disabled (run to max episodes)")
        print(f"   Log Interval: {LOG_INTERVAL}")
        print(f"   Checkpoint Interval: {checkpoint_interval}")
        print("="*80)
    
    # Progress bar
    pbar = tqdm(range(start_episode, n_episodes), desc="Training", disable=not verbose, initial=start_episode, total=n_episodes)
    
    for episode in pbar:
        state = mdp.reset()
        episode_profit = 0
        episode_return = 0
        step_count = 0
        
        while True:
            # Select action
            if is_dqn:
                action = agent.select_action(state)
            else:
                # Epsilon-greedy cho tabular
                if np.random.random() < epsilon:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(Q[state])
            
            next_state, reward, done, info = mdp.step(action)
            
            # Update Q-value
            if is_dqn:
                agent.update_q_value(state, action, reward, next_state, done)
            else:
                # Tabular Q-learning update
                current_q = Q[state][action]
                next_max_q = 0 if done else np.max(Q[next_state])
                Q[state][action] = current_q + alpha * (reward + gamma * next_max_q - current_q)
            
            episode_profit += reward
            episode_return += (gamma ** step_count) * reward
            state = next_state
            step_count += 1
            
            if done:
                break
        
        # Decay epsilon
        if is_dqn:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            current_epsilon = agent.epsilon
        else:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            current_epsilon = epsilon
        
        # Track history
        history['episodes'].append(episode)
        history['profits'].append(episode_profit)
        history['epsilon'].append(current_epsilon)
        history['returns'].append(episode_return)
        
        # Check for improvement
        if episode_profit > best_profit + min_improvement:
            best_profit = episode_profit
            patience_counter = 0
            history['best_episode'] = episode
            history['best_profit'] = best_profit
        else:
            patience_counter += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Profit': f'{episode_profit:.4f}',
            'Best': f'{best_profit:.4f}',
            'ε': f'{current_epsilon:.3f}',
            'Patience': f'{(patience - patience_counter) if early_stopping else "-"}'
        })
        
        # Auto-save checkpoint
        if checkpoint_interval > 0 and (episode + 1) % checkpoint_interval == 0:
            try:
                # Convert Q to regular dict if defaultdict
                Q_to_save = dict(Q) if isinstance(Q, dict) else Q
                
                checkpoint_manager.auto_save_checkpoint(
                    Q=Q_to_save,
                    episode=episode + 1,
                    history=history,
                    mdp_state={
                        'balance': mdp.balance,
                        'holdings': getattr(mdp, 'holdings', 0),
                        'current_step': getattr(mdp, 'current_step', 0)
                    },
                    metadata={
                        'patience_counter': patience_counter,
                        'epsilon': current_epsilon,
                        'best_profit': best_profit
                    },
                    save_interval=checkpoint_interval,
                    keep_last_n=5
                )
            except Exception as e:
                if verbose:
                    print(f"⚠️ Checkpoint save skipped: {e}")
        
        # Early stopping (optional)
        if early_stopping and patience_counter >= patience and episode > 200:
            if verbose:
                print(f"\n✋ Early stopping at episode {episode + 1} (no improvement for {patience} episodes)")
            history['converged'] = True
            break
        
        # Periodic logging
        if verbose and (episode + 1) % LOG_INTERVAL == 0:
            avg_profit = np.mean(history['profits'][-100:])
            print(f"\nEpisode {episode + 1:5d}/{n_episodes} | "
                  f"Profit: {episode_profit:.4f} | "
                  f"Avg(100): {avg_profit:.4f} | "
                  f"Best: {best_profit:.4f} | "
                  f"ε: {current_epsilon:.3f}")
    
    pbar.close()
    
    if verbose:
        print("="*80)
        print("✅ Training completed!")
        print(f"   Total Episodes: {len(history['episodes'])}")
        print(f"   Best Profit: {history['best_profit']:.4f} (Episode {history['best_episode']})")
        # Final epsilon depends on mode
        final_eps = history['epsilon'][-1] if len(history['epsilon']) > 0 else None
        if final_eps is not None:
            print(f"   Final Epsilon: {final_eps:.3f}")
        print(f"   Converged: {history['converged']}")
        print("="*80)
    
    return history


def train_advanced_with_experience_replay(agent, mdp, n_episodes=None, 
                                          batch_size=32, memory_size=10000,
                                          patience=None, verbose=True):
    """
    Training với experience replay - phù hợp cho neural network
    
    Args:
        agent: Q-Learning agent (with neural network)
        mdp: MDP environment
        n_episodes: Số episodes
        batch_size: Batch size cho training
        memory_size: Kích thước replay buffer
        patience: Early stopping patience
        verbose: In thông tin
    
    Returns:
        Training history
    """
    if n_episodes is None:
        n_episodes = TRAINING_EPISODES
    if patience is None:
        patience = EARLY_STOPPING_PATIENCE
    
    from collections import deque
    
    replay_buffer = deque(maxlen=memory_size)
    
    history = {
        'episodes': [],
        'profits': [],
        'epsilon': [],
        'losses': [],
        'best_episode': 0,
        'best_profit': -np.inf
    }
    
    best_profit = -np.inf
    patience_counter = 0
    
    if verbose:
        print("\n🧠 Advanced Training with Experience Replay")
        print("="*80)
        print(f"   Device: {DEVICE}")
        print(f"   Memory Size: {memory_size}")
        print(f"   Batch Size: {batch_size}")
        print("="*80)
    
    pbar = tqdm(range(n_episodes), desc="Training", disable=not verbose)
    
    for episode in pbar:
        state = mdp.reset()
        episode_profit = 0
        episode_loss = 0
        loss_count = 0
        
        # Lấy state features
        state_features = mdp._get_state_features(state)
        
        while True:
            # Select action
            action = agent.select_action(state_features)
            
            # Execute action
            next_state, reward, done, info = mdp.step(action)
            next_state_features = mdp._get_state_features(next_state)
            
            # Store experience
            replay_buffer.append((state_features, action, reward, next_state_features, done))
            
            episode_profit += reward
            
            # Train from replay buffer
            if len(replay_buffer) >= batch_size:
                # Sample batch
                indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch = [replay_buffer[i] for i in indices]
                
                # Update agent
                for s, a, r, ns, d in batch:
                    agent.update_q_value(s, a, r, ns, d)
                    loss_count += 1
            
            state = next_state
            state_features = next_state_features
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track history
        history['episodes'].append(episode)
        history['profits'].append(episode_profit)
        history['epsilon'].append(agent.epsilon)
        history['losses'].append(episode_loss / max(loss_count, 1))
        
        # Check improvement
        if episode_profit > best_profit:
            best_profit = episode_profit
            patience_counter = 0
            history['best_episode'] = episode
            history['best_profit'] = best_profit
        else:
            patience_counter += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Profit': f'{episode_profit:.4f}',
            'Best': f'{best_profit:.4f}',
            'ε': f'{agent.epsilon:.3f}'
        })
        
        # Early stopping
        if patience_counter >= patience and episode > 200:
            if verbose:
                print(f"\n✋ Early stopping at episode {episode + 1}")
            break
    
    pbar.close()
    
    if verbose:
        print("="*80)
        print("✅ Training completed!")
        print(f"   Best Profit: {history['best_profit']:.4f}")
        print("="*80)
    
    return history


def plot_training_history(history, save_path=None):
    """
    Vẽ biểu đồ training history
    
    Args:
        history: Training history dict
        save_path: Đường dẫn lưu plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Profit over episodes
    axes[0, 0].plot(history['episodes'], history['profits'], alpha=0.6, label='Episode Profit')
    if len(history['profits']) > 50:
        window = 50
        moving_avg = pd.Series(history['profits']).rolling(window=window).mean()
        axes[0, 0].plot(history['episodes'], moving_avg, 'r-', linewidth=2, label=f'MA({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Profit')
    axes[0, 0].set_title('Training Profit Over Episodes')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Epsilon decay
    axes[0, 1].plot(history['episodes'], history['epsilon'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon')
    axes[0, 1].set_title('Epsilon Decay')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Returns (if available)
    if 'returns' in history:
        axes[1, 0].plot(history['episodes'], history['returns'], alpha=0.6, label='Episode Return')
        if len(history['returns']) > 50:
            window = 50
            moving_avg = pd.Series(history['returns']).rolling(window=window).mean()
            axes[1, 0].plot(history['episodes'], moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Return')
        axes[1, 0].set_title('Discounted Returns Over Episodes')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss (if available)
    if 'losses' in history and len(history['losses']) > 0:
        axes[1, 1].plot(history['episodes'], history['losses'], 'purple', alpha=0.6)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Profit distribution
        axes[1, 1].hist(history['profits'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].set_xlabel('Profit')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Profit Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved to {save_path}")
    
    if SAVE_PLOTS:
        os.makedirs(PLOT_DIR, exist_ok=True)
        default_path = os.path.join(PLOT_DIR, 'training_history.png')
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history also saved to {default_path}")
    
    plt.show()
    
    return fig
