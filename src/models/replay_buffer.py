"""
Experience Replay Buffer for DQN
Stores and samples (state, action, reward, next_state, done) transitions
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer cho DQN
    
    Lưu trữ transitions (s, a, r, s', done) và sample random batches
    để break correlation giữa consecutive samples
    """
    
    def __init__(self, capacity: int = 100000, seed: int = None):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to buffer
        
        Args:
            state: Current state (numpy array)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state (numpy array)
            done: Episode done flag (bool)
        """
        # Convert to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Store transition
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each as numpy array with shape (batch_size, ...)
        """
        # Sample random batch
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def sample_tensors(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
        """
        Sample batch and convert to PyTorch tensors
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to put tensors on ('cpu' or 'cuda')
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        states, actions, rewards, next_states, dones = self.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch"""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.position = 0
    
    def save(self, filepath: str):
        """Save buffer to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"✅ Replay buffer saved to: {filepath} ({len(self)} transitions)")
    
    def load(self, filepath: str):
        """Load buffer from disk"""
        import pickle
        with open(filepath, 'rb') as f:
            transitions = pickle.load(f)
        
        self.buffer.clear()
        for transition in transitions:
            self.buffer.append(transition)
        
        print(f"✅ Replay buffer loaded from: {filepath} ({len(self)} transitions)")
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0
            }
        
        # Extract rewards for statistics
        rewards = [t[2] for t in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity * 100,
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'std_reward': np.std(rewards)
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER)
    
    Sample transitions with higher TD-error more frequently
    Helps focus on important experiences
    
    Reference: Schaul et al. (2015) - Prioritized Experience Replay
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, seed: int = None):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Increment beta towards 1.0 during training
            seed: Random seed
        """
        super().__init__(capacity, seed)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        # Store priorities
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Add transition with maximum priority"""
        super().push(state, action, reward, next_state, done)
        
        # New transitions get max priority
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch based on priorities
        
        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        
        # Get transitions
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Convert to arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)
        
        # Increment beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on TD-errors
        
        Args:
            indices: Indices of sampled transitions
            td_errors: TD-errors (absolute values)
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


# Test code
if __name__ == "__main__":
    print("="*80)
    print("TESTING REPLAY BUFFER")
    print("="*80)
    
    # Test standard replay buffer
    print("\n1. Testing Standard Replay Buffer...")
    buffer = ReplayBuffer(capacity=1000, seed=42)
    
    # Add some transitions
    for i in range(100):
        state = np.random.randn(7)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(7)
        done = i % 10 == 0
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"   Buffer size: {len(buffer)}")
    print(f"   Is ready for batch=32? {buffer.is_ready(32)}")
    
    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"   Sampled batch shapes:")
    print(f"      States: {states.shape}")
    print(f"      Actions: {actions.shape}")
    print(f"      Rewards: {rewards.shape}")
    
    # Get stats
    stats = buffer.get_stats()
    print(f"   Buffer stats:")
    print(f"      Size: {stats['size']}")
    print(f"      Utilization: {stats['utilization']:.1f}%")
    print(f"      Avg reward: {stats['avg_reward']:.4f}")
    
    # Test prioritized replay buffer
    print("\n2. Testing Prioritized Replay Buffer...")
    per_buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
    
    # Add transitions
    for i in range(100):
        state = np.random.randn(7)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(7)
        done = False
        
        per_buffer.push(state, action, reward, next_state, done)
    
    print(f"   PER Buffer size: {len(per_buffer)}")
    
    # Sample with priorities
    result = per_buffer.sample(32)
    states, actions, rewards, next_states, dones, indices, weights = result
    print(f"   Sampled with priorities:")
    print(f"      Indices: {indices[:5]}...")
    print(f"      Weights: {weights[:5]}")
    
    print("\n✅ All tests passed!")
    print("="*80)
