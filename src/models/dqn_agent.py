"""
DQN Agent cho cryptocurrency trading
Integrates DQN network, replay buffer, và training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional

try:
    from .dqn_network import DQNetwork, DuelingDQN
    from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
except ImportError:
    # For standalone testing
    from dqn_network import DQNetwork, DuelingDQN
    from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent với experience replay và target network
    
    Key Features:
    - Experience replay để break correlation
    - Target network để stabilize training  
    - Epsilon-greedy exploration
    - GPU acceleration
    - Paper methodology (trend-based rewards đã có trong MDP)
    """
    
    def __init__(self,
                 state_dim: int = 7,
                 action_dim: int = 3,
                 hidden_dim_1: int = 128,
                 hidden_dim_2: int = 64,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 100000,
                 batch_size: int = 64,
                 target_update_frequency: int = 1000,
                 use_dueling: bool = False,
                 use_prioritized: bool = False,
                 device: str = None,
                 seed: int = None):
        """
        Initialize DQN Agent
        
        Args:
            state_dim: State space dimension (default: 7)
            action_dim: Action space dimension (default: 3)
            hidden_dim_1: First hidden layer size
            hidden_dim_2: Second hidden layer size
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            target_update_frequency: Steps between target network updates
            use_dueling: Use Dueling DQN architecture
            use_prioritized: Use Prioritized Experience Replay
            device: 'cuda' or 'cpu' (auto-detect if None)
            seed: Random seed
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing DQN Agent on device: {self.device}")
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Create networks
        if use_dueling:
            self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dim_1).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim_1).to(self.device)
            print("   Using Dueling DQN architecture")
        else:
            self.policy_net = DQNetwork(state_dim, action_dim, hidden_dim_1, hidden_dim_2).to(self.device)
            self.target_net = DQNetwork(state_dim, action_dim, hidden_dim_1, hidden_dim_2).to(self.device)
            print("   Using Standard DQN architecture")
        
        # Copy policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net always in eval mode
        
        # Optimizer và loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        if use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, seed=seed)
            self.use_prioritized = True
            print("   Using Prioritized Experience Replay")
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)
            self.use_prioritized = False
            print("   Using Standard Experience Replay")
        
        # Training counters
        self.steps = 0
        self.episodes = 0
        self.update_counter = 0
        self.total_loss = 0.0
        
        print(f"DQN Agent initialized")
        print(f"   Parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")
    
    def get_q_value(self, state) -> np.ndarray:
        """
        Get Q-values for a state (for compatibility with validation script)
        
        Args:
            state: State vector (numpy array or torch tensor)
        
        Returns:
            q_values: Numpy array of shape (action_dim,)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
        
        return q_values.cpu().numpy()
    
    def select_action(self, state, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            epsilon: Exploration rate (use self.epsilon if None)
        
        Returns:
            action: Selected action (0=Buy, 1=Sell, 2=Hold)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Random exploration
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Greedy exploitation
        return self.policy_net.get_action(state, epsilon=0.0)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step (if buffer has enough samples)
        
        Returns:
            loss: Training loss (0.0 if not ready to train)
        """
        # Check if buffer is ready
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0
        
        # Sample batch
        if self.use_prioritized:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s,a) - current Q-values
        current_q_values = self.policy_net(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values: r + gamma * max Q(s',a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        if self.use_prioritized:
            td_errors = (current_q - target_q).detach().cpu().numpy()
            loss = (weights * (current_q - target_q).pow(2)).mean()
            
            # Update priorities
            self.replay_buffer.update_priorities(indices, td_errors)
        else:
            loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping để prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update counters
        self.update_counter += 1
        self.total_loss += loss.item()
        
        # Update target network periodically
        if self.update_counter % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes += 1
    
    def save_checkpoint(self, filepath: str, episode: int, metadata: dict = None):
        """
        Save complete checkpoint
        
        Args:
            filepath: Path to save checkpoint
            episode: Current episode number
            metadata: Optional metadata dict with training info (training_time, best_profit, etc.)
        """
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'update_counter': self.update_counter,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'target_update_frequency': self.target_update_frequency
            }
        }
        
        # Add metadata if provided (training_time, best_profit, etc.)
        if metadata:
            checkpoint['metadata'] = metadata
            # Also add best_profit to top level for backward compatibility
            if 'best_profit' in metadata:
                checkpoint['best_profit'] = metadata['best_profit']
        
        torch.save(checkpoint, filepath)
        print(f"✅ DQN checkpoint saved: {filepath}")
        print(f"   Episode: {episode}, Epsilon: {self.epsilon:.4f}, Updates: {self.update_counter:,}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.update_counter = checkpoint['update_counter']
        self.episodes = checkpoint['episode']
        
        print(f"DQN checkpoint loaded: {filepath}")
        print(f"   Episode: {self.episodes}, Epsilon: {self.epsilon:.4f}, Updates: {self.update_counter:,}")
        
        return checkpoint
    
    def get_stats(self) -> dict:
        """Get training statistics"""
        stats = {
            'episodes': self.episodes,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            'avg_loss': self.total_loss / max(1, self.update_counter),
            'buffer_size': len(self.replay_buffer),
            'buffer_utilization': len(self.replay_buffer) / self.replay_buffer.capacity * 100
        }
        
        # Add buffer stats
        buffer_stats = self.replay_buffer.get_stats()
        stats.update(buffer_stats)
        
        return stats


# Test code
if __name__ == "__main__":
    print("="*80)
    print("TESTING DQN AGENT")
    print("="*80)
    
    # Test agent creation
    print("\n1. Creating DQN Agent...")
    agent = DQNAgent(
        state_dim=7,
        action_dim=3,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        batch_size=32,
        buffer_capacity=10000,
        use_dueling=False,
        use_prioritized=False
    )
    
    # Test action selection
    print("\n2. Testing action selection...")
    state = np.random.randn(7)
    action = agent.select_action(state, epsilon=0.5)
    print(f"   Selected action: {action}")
    
    # Test Q-value retrieval
    q_values = agent.get_q_value(state)
    print(f"   Q-values: {q_values}")
    
    # Test storing transitions and training
    print("\n3. Testing training loop...")
    for i in range(100):
        state = np.random.randn(7)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(7)
        done = i % 10 == 0
        
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"   Buffer size: {len(agent.replay_buffer)}")
    
    # Train a few steps
    losses = []
    for _ in range(10):
        loss = agent.train_step()
        losses.append(loss)
    
    print(f"   Training losses: {losses[:5]}")
    print(f"   Avg loss: {np.mean(losses):.4f}")
    
    # Test epsilon decay
    print("\n4. Testing epsilon decay...")
    print(f"   Initial epsilon: {agent.epsilon:.4f}")
    agent.update_epsilon()
    print(f"   After 1 episode: {agent.epsilon:.4f}")
    
    # Test stats
    print("\n5. Agent statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\nAll tests passed!")
    print("="*80)
