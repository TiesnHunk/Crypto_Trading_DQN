"""
Deep Q-Network (DQN) implementation for cryptocurrency trading
Giải quyết state collapse issue của Tabular Q-Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNetwork(nn.Module):
    """
    Deep Q-Network cho trading
    
    Input: Continuous state vector [position, portfolio, rsi, macd, trend, bb, volatility]
    Output: Q-values for [Buy, Sell, Hold]
    
    Architecture:
        Input (7) → FC(128) → ReLU → Dropout → FC(64) → ReLU → Dropout → Output (3)
    """
    
    def __init__(self, state_dim: int = 7, action_dim: int = 3, 
                 hidden_dim_1: int = 128, hidden_dim_2: int = 64,
                 dropout_rate: float = 0.2):
        """
        Initialize DQN
        
        Args:
            state_dim: Number of state features (default: 7)
            action_dim: Number of actions (default: 3 - Buy/Sell/Hold)
            hidden_dim_1: First hidden layer size (default: 128)
            hidden_dim_2: Second hidden layer size (default: 64)
            dropout_rate: Dropout rate for regularization (default: 0.2)
        """
        super(DQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim_1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim_2, action_dim)
        
        # Initialize weights với Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            q_values: Tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        # Handle single state or batch
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        x = self.fc1(state)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        q_values = self.fc3(x)
        
        # Remove batch dimension if input was single state
        if q_values.shape[0] == 1:
            q_values = q_values.squeeze(0)
        
        return q_values
    
    def get_action(self, state, epsilon: float = 0.0):
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: State tensor or numpy array
            epsilon: Exploration rate (0.0 = pure greedy)
        
        Returns:
            action: Selected action (0=Buy, 1=Sell, 2=Hold)
        """
        # Random exploration
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Convert state to tensor if needed and move to correct device
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Move to network's device
        device = next(self.parameters()).device
        state = state.to(device)
        
        # Greedy action selection
        with torch.no_grad():
            q_values = self.forward(state)
            action = torch.argmax(q_values).item()
        
        return action
    
    def save(self, filepath: str):
        """Save network weights"""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, filepath)
        print(f"✅ DQN saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load network weights"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"✅ DQN loaded from: {filepath}")
        return self


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture (Optional - better performance)
    
    Separates Q-value into:
    - Value function V(s): How good is this state?
    - Advantage function A(s,a): How much better is action a?
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """
    
    def __init__(self, state_dim: int = 7, action_dim: int = 3,
                 hidden_dim: int = 128, dropout_rate: float = 0.2):
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass with dueling architecture
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Extract features
        features = self.feature(state)
        
        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        if q_values.shape[0] == 1:
            q_values = q_values.squeeze(0)
        
        return q_values
    
    def get_action(self, state, epsilon: float = 0.0):
        """Get action using epsilon-greedy"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Move to network's device
        device = next(self.parameters()).device
        state = state.to(device)
        
        with torch.no_grad():
            q_values = self.forward(state)
            action = torch.argmax(q_values).item()
        
        return action
    
    def save(self, filepath: str):
        """Save network"""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, filepath)
        print(f"✅ Dueling DQN saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load network"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"✅ Dueling DQN loaded from: {filepath}")
        return self


# Test code
if __name__ == "__main__":
    print("="*80)
    print("TESTING DQN NETWORK")
    print("="*80)
    
    # Test standard DQN
    print("\n1. Testing Standard DQN...")
    dqn = DQNetwork(state_dim=7, action_dim=3)
    print(f"   Network parameters: {sum(p.numel() for p in dqn.parameters()):,}")
    
    # Test single state
    state = torch.randn(7)
    q_values = dqn(state)
    print(f"   Single state Q-values: {q_values}")
    
    # Test batch
    batch_states = torch.randn(32, 7)
    batch_q_values = dqn(batch_states)
    print(f"   Batch Q-values shape: {batch_q_values.shape}")
    
    # Test action selection
    action = dqn.get_action(state, epsilon=0.0)
    print(f"   Selected action (greedy): {action}")
    
    # Test Dueling DQN
    print("\n2. Testing Dueling DQN...")
    dueling_dqn = DuelingDQN(state_dim=7, action_dim=3)
    print(f"   Network parameters: {sum(p.numel() for p in dueling_dqn.parameters()):,}")
    
    q_values_dueling = dueling_dqn(state)
    print(f"   Single state Q-values: {q_values_dueling}")
    
    print("\n✅ All tests passed!")
    print("="*80)
