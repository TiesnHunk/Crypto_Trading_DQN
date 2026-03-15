"""
Q-Learning Agent with GPU Support
Module implement thuật toán Q-Learning có hỗ trợ GPU
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, List
import pickle
import os
import torch
import torch.nn as nn

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.config import DEVICE, USE_GPU, Q_LEARNING_PARAMS


class QLearningAgent:
    """
    Class implement Q-Learning agent cho trading với GPU support
    """
    
    def __init__(self, state_dim: int = 6, n_actions: int = 3, alpha: float = 0.1, 
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01, discrete: bool = True, n_bins: int = 10,
                 use_gpu: bool = True, device=None):
        """
        Khởi tạo Q-Learning Agent với GPU support
        
        Args:
            state_dim: Chiều của state space
            n_actions: Số lượng actions (3: Hold, Buy, Sell)
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Epsilon cho epsilon-greedy
            epsilon_decay: Tỷ lệ giảm epsilon
            epsilon_min: Giá trị epsilon tối thiểu
            discrete: Có sử dụng discretization không
            n_bins: Số bins cho discretization
            use_gpu: Sử dụng GPU nếu có
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discrete = discrete
        self.n_bins = n_bins
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = device if device else DEVICE
        
        print(f"🤖 Q-Learning Agent initialized")
        print(f"   Device: {self.device}")
        print(f"   State dim: {state_dim}, Actions: {n_actions}")
        print(f"   Discrete: {discrete}, Bins: {n_bins if discrete else 'N/A'}")
        
        # Q-table: Nếu dùng discrete, dùng dict. Nếu dùng continuous, dùng neural network
        if discrete:
            self.Q = defaultdict(lambda: np.zeros(n_actions))
        else:
            # Neural network approximation cho continuous state
            # ✅ MAJOR FIX: Tăng network size để học được state differentiation
            # Từ [128, 64, 32] → [256, 128, 64, 32] với dropout hợp lý
            self.Q_network = QNetwork(state_dim, n_actions, hidden_dims=[256, 128, 64, 32]).to(self.device)
            # ✅ FIX Q-NETWORK COLLAPSE: Better initialization + weight decay
            self._initialize_network()  # Custom initialization
            # ✅ GPU FIX: Set initial mode (training mode)
            self.Q_network.train()
            # ✅ GPU FIX: Lower learning rate for GPU stability (GPU thường cần LR thấp hơn)
            gpu_lr = alpha * 0.5 if self.device.type == 'cuda' else alpha
            self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=gpu_lr, weight_decay=1e-4)
            self.criterion = nn.MSELoss()
            
            # ✅ Batch buffer cho experience replay (tăng GPU utilization)
            self.batch_buffer = []
            self.batch_size = 16  # ✅ V4: Giảm từ 32 → 16 để GPU update thường xuyên hơn
            self.min_batch_size = 8  # ✅ V4: Mini-batch update khi >= 8 samples
            self.update_counter = 0
        
        # Lịch sử training
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'epsilon': [],
            'avg_reward': [],
            'profits': []
        }
    
    def __getstate__(self):
        """✅ FIX PICKLE: Convert defaultdict to dict for pickling"""
        state = self.__dict__.copy()
        # Convert defaultdict Q to regular dict if it exists
        if 'Q' in state and hasattr(state.get('Q'), '__missing__'):
            state['Q'] = dict(state['Q'])
        return state
    
    def __setstate__(self, state):
        """✅ FIX PICKLE: Restore defaultdict from dict after unpickling"""
        self.__dict__.update(state)
        # Restore defaultdict if discrete Q-learning
        if state.get('discrete', False) and 'Q' in state and isinstance(state['Q'], dict):
            from collections import defaultdict
            n_actions = state.get('n_actions', 3)
            self.Q = defaultdict(lambda: np.zeros(n_actions), state['Q'])
    
    def _initialize_network(self):
        """
        ✅ FIX Q-NETWORK COLLAPSE: Better initialization để tránh Q-values collapse
        Sử dụng Xavier/He initialization cho weights và zero init cho biases
        """
        for layer in self.Q_network.modules():
            if isinstance(layer, nn.Linear):
                # Xavier initialization cho weights
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                # Zero initialization cho biases (hoặc small random)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        print(f"   ✅ Network initialized with Xavier init")
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Chuyển continuous state sang discrete state
        
        Args:
            state: Continuous state vector
        
        Returns:
            Discrete state tuple
        """
        discrete_state = []
        for value in state:
            # Clip giá trị về [0, 1] nếu cần
            value = np.clip(value, 0, 1)
            bin_idx = int(value * (self.n_bins - 1))
            bin_idx = min(bin_idx, self.n_bins - 1)
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def get_q_value(self, state: np.ndarray, action: int = None):
        """
        Lấy Q-value cho state
        
        Args:
            state: State vector
            action: Action (nếu None thì trả về Q-values cho tất cả actions)
        
        Returns:
            Q-value(s)
        """
        if self.discrete:
            discrete_state = self.discretize_state(state)
            if action is not None:
                return self.Q[discrete_state][action]
            else:
                return self.Q[discrete_state]
        else:
            # Neural network approximation
            # ✅ GPU FIX: Set eval mode để disable dropout khi inference
            self.Q_network.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.Q_network(state_tensor).cpu().numpy()[0]
                if action is not None:
                    return q_values[action]
                else:
                    return q_values
    
    def select_action(self, state: np.ndarray, epsilon: float = None) -> int:
        """
        Chọn action theo epsilon-greedy policy
        
        Args:
            state: State hiện tại
            epsilon: Epsilon value (nếu None thì dùng self.epsilon)
        
        Returns:
            Action index
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.get_q_value(state)
            return int(np.argmax(q_values))
    
    def update_q_value(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool):
        """
        Cập nhật Q-value
        
        Args:
            state: State hiện tại
            action: Action đã thực hiện
            reward: Reward nhận được
            next_state: State tiếp theo
            done: Episode đã kết thúc chưa
        """
        if self.discrete:
            # Discrete Q-table update
            discrete_state = self.discretize_state(state)
            discrete_next_state = self.discretize_state(next_state)
            
            current_q = self.Q[discrete_state][action]
            
            if done:
                target_q = reward
            else:
                next_max_q = np.max(self.Q[discrete_next_state])
                target_q = reward + self.gamma * next_max_q
            
            self.Q[discrete_state][action] = current_q + self.alpha * (target_q - current_q)
        else:
            # Neural network update với batch processing (tăng GPU utilization)
            # ✅ Add to batch buffer
            self.batch_buffer.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            # ✅ V4: Update khi đủ batch size hoặc mini-batch size
            # Update thường xuyên hơn để GPU luôn được sử dụng
            if len(self.batch_buffer) >= self.batch_size:
                self._update_batch()
                self.update_counter += 1
            elif len(self.batch_buffer) >= self.min_batch_size:
                # ✅ Mini-batch update để GPU không idle khi episodes ngắn
                self._update_batch()
                self.update_counter += 1
    
    def _update_batch(self):
        """
        Update network với batch của experiences (tăng GPU utilization)
        """
        if len(self.batch_buffer) == 0:
            return
        
        # Prepare batch tensors - ✅ Fix: Convert list to numpy array trước khi tạo tensor
        states = torch.FloatTensor(np.array([exp['state'] for exp in self.batch_buffer])).to(self.device)
        actions = torch.LongTensor(np.array([exp['action'] for exp in self.batch_buffer])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp['reward'] for exp in self.batch_buffer])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp['next_state'] for exp in self.batch_buffer])).to(self.device)
        dones = torch.BoolTensor(np.array([exp['done'] for exp in self.batch_buffer])).to(self.device)
        
        # ✅ GPU FIX: Set train mode để enable dropout khi training
        self.Q_network.train()
        
        # Current Q-values (batch_size x n_actions)
        q_values = self.Q_network(states)  # Shape: [batch_size, n_actions]
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]
        
        # Target Q-values (use eval mode để có Q-values ổn định)
        self.Q_network.eval()
        with torch.no_grad():
            next_q_values = self.Q_network(next_states)  # Shape: [batch_size, n_actions]
            next_max_q = next_q_values.max(1)[0]  # Shape: [batch_size]
            
            # Target = reward + gamma * max(Q(s', a')) if not done, else reward
            target_q = rewards + (~dones).float() * self.gamma * next_max_q  # Shape: [batch_size]
        
        # Switch back to train mode for loss computation
        self.Q_network.train()
        
        # Compute loss (both shapes are [batch_size])
        loss = self.criterion(current_q, target_q)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        # ✅ GPU FIX: Gradient clipping để tránh gradient explosion trên GPU
        torch.nn.utils.clip_grad_norm_(self.Q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Clear buffer
        self.batch_buffer.clear()
    
    def flush_batch(self):
        """
        Flush batch buffer (update với samples còn lại)
        """
        if not self.discrete and len(self.batch_buffer) > 0:
            # ✅ GPU FIX: Ensure train mode
            self.Q_network.train()
            self._update_batch()
    
    def decay_epsilon(self):
        """Giảm epsilon"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """
        Lưu agent
        
        Args:
            filepath: Đường dẫn file
        """
        save_dict = {
            'state_dim': self.state_dim,
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'discrete': self.discrete,
            'n_bins': self.n_bins,
            'training_history': self.training_history
        }
        
        if self.discrete:
            save_dict['Q'] = dict(self.Q)
        else:
            save_dict['Q_network_state'] = self.Q_network.state_dict()
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✅ Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent
        
        Args:
            filepath: Đường dẫn file
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.state_dim = save_dict['state_dim']
        self.n_actions = save_dict['n_actions']
        self.alpha = save_dict['alpha']
        self.gamma = save_dict['gamma']
        self.epsilon = save_dict['epsilon']
        self.epsilon_decay = save_dict['epsilon_decay']
        self.epsilon_min = save_dict['epsilon_min']
        self.discrete = save_dict['discrete']
        self.n_bins = save_dict['n_bins']
        self.training_history = save_dict['training_history']
        
        if self.discrete:
            self.Q = defaultdict(lambda: np.zeros(self.n_actions), save_dict['Q'])
        else:
            self.Q_network.load_state_dict(save_dict['Q_network_state'])
        
        print(f"✅ Agent loaded from {filepath}")


class QNetwork(nn.Module):
    """
    Neural network cho Q-value approximation
    ✅ FIX Q-NETWORK COLLAPSE: Tăng capacity để học được state differentiation
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dims: List[int] = [256, 128, 64, 32]):
        """
        Khởi tạo Q-Network
        
        Args:
            state_dim: Chiều của state
            n_actions: Số lượng actions
            hidden_dims: Số neurons trong các hidden layers
                         ✅ MAJOR FIX: Tăng lên [256, 128, 64, 32] để học được state differentiation
        """
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            # ✅ Dropout hợp lý: 0.25 để balance learning và regularization
            layers.append(nn.Dropout(0.25))  # Balanced dropout
            input_dim = hidden_dim
        
        # Output layer (không có dropout)
        layers.append(nn.Linear(input_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: State tensor
        
        Returns:
            Q-values
        """
        return self.network(state)
