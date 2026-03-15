"""
DQN + PSO + LSTM for Trading
Kết hợp Deep Q-Network với LSTM backbone và PSO optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
import copy


class DQNLSTMNetwork(nn.Module):
    """
    DQN Network với LSTM backbone
    Input: Sequence of states
    Output: Q-values for each action
    """
    
    def __init__(self,
                 input_size: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 action_dim: int = 3,
                 dropout: float = 0.2):
        """
        Args:
            input_size: Số features đầu vào
            hidden_size: LSTM hidden size
            num_layers: Số LSTM layers
            action_dim: Số actions (3: Buy/Sell/Hold)
            dropout: Dropout rate
        """
        super(DQNLSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Q-value head
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Q-values (batch_size, action_dim)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Lấy output của timestep cuối cùng
        lstm_out = lstm_out[:, -1, :]
        
        # Q-value head
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        q_values = self.fc3(out)
        
        return q_values


class DQNLSTMAgent:
    """
    DQN Agent với LSTM network
    """
    
    def __init__(self,
                 input_size: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 sequence_length: int = 24,
                 action_dim: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 target_update_frequency: int = 100,
                 device: str = None):
        """
        Initialize DQN-LSTM Agent
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Networks
        self.policy_net = DQNLSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            action_dim=action_dim
        ).to(self.device)
        
        self.target_net = DQNLSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            action_dim=action_dim
        ).to(self.device)
        
        # Copy weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.buffer = []
        self.buffer_capacity = buffer_capacity
        
        # Counters
        self.steps = 0
        self.episodes = 0
        
    def prepare_sequence(self, data: np.ndarray, index: int) -> np.ndarray:
        """
        Chuẩn bị sequence từ data tại index
        """
        start_idx = max(0, index - self.sequence_length + 1)
        end_idx = index + 1
        
        sequence = data[start_idx:end_idx]
        
        # Pad nếu thiếu
        if len(sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(sequence), self.input_size))
            sequence = np.vstack([padding, sequence])
        
        return sequence
    
    def select_action(self, state_sequence: np.ndarray, epsilon: float = None) -> int:
        """
        Select action using epsilon-greedy
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        self.policy_net.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def store_transition(self, state_seq, action, reward, next_state_seq, done):
        """
        Store transition in replay buffer
        """
        if len(self.buffer) >= self.buffer_capacity:
            self.buffer.pop(0)
        
        self.buffer.append((state_seq, action, reward, next_state_seq, done))
    
    def train_step(self):
        """
        Train one step
        """
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        # Current Q-values
        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """
        Decay epsilon
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def predict_action(self, state_sequence: np.ndarray) -> int:
        """
        Predict action (greedy)
        """
        return self.select_action(state_sequence, epsilon=0.0)
    
    def save_checkpoint(self, filepath: str, episode: int = 0, best_reward: float = 0):
        """
        Save checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'episode': episode,
            'best_reward': best_reward,
            'input_size': self.input_size,
            'hidden_size': self.policy_net.hidden_size,
            'num_layers': self.policy_net.num_layers,
            'sequence_length': self.sequence_length,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_frequency': self.target_update_frequency
        }
        torch.save(checkpoint, filepath)
        print(f"   💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        
        print(f"   ✅ Checkpoint loaded: {filepath}")
        print(f"   Episode: {checkpoint.get('episode', 0)}")
        print(f"   Best reward: {checkpoint.get('best_reward', 0):.2f}")
        
        return checkpoint


class Particle:
    """
    Particle cho PSO optimization
    """
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        self.bounds = bounds
        self.position = {}
        self.velocity = {}
        self.best_position = {}
        self.best_fitness = -float('inf')
        self.fitness = -float('inf')
        
        # Khởi tạo position và velocity ngẫu nhiên
        for param, (min_val, max_val) in bounds.items():
            self.position[param] = np.random.uniform(min_val, max_val)
            range_val = max_val - min_val
            self.velocity[param] = np.random.uniform(-range_val/10, range_val/10)
        
        self.best_position = copy.deepcopy(self.position)
    
    def update_velocity(self, global_best_position: Dict, w: float = 0.7, 
                        c1: float = 1.5, c2: float = 1.5):
        """
        Update velocity
        """
        for param in self.position.keys():
            r1 = np.random.random()
            r2 = np.random.random()
            
            cognitive = c1 * r1 * (self.best_position[param] - self.position[param])
            social = c2 * r2 * (global_best_position[param] - self.position[param])
            
            self.velocity[param] = w * self.velocity[param] + cognitive + social
    
    def update_position(self):
        """
        Update position
        """
        for param in self.position.keys():
            self.position[param] += self.velocity[param]
            
            # Clamp to bounds
            min_val, max_val = self.bounds[param]
            self.position[param] = np.clip(self.position[param], min_val, max_val)
    
    def get_integer_params(self) -> Dict:
        """
        Convert to integer params
        """
        params = {}
        for key, value in self.position.items():
            if key in ['hidden_size', 'num_layers', 'sequence_length', 
                      'batch_size', 'target_update_frequency']:
                params[key] = int(round(value))
            else:
                params[key] = value
        return params


class DQNPSOLSTMOptimizer:
    """
    PSO Optimizer cho DQN-LSTM hyperparameters
    """
    
    def __init__(self,
                 n_particles: int = 5,
                 max_iterations: int = 10,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        Initialize PSO optimizer
        """
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.global_best_position = None
        self.global_best_fitness = -float('inf')
        self.fitness_history = []
    
    def evaluate_particle(self, particle, env, input_size: int, 
                         n_episodes: int = 3):
        """
        Evaluate một particle
        """
        params = particle.get_integer_params()
        
        try:
            # Create agent
            agent = DQNLSTMAgent(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                sequence_length=params['sequence_length'],
                learning_rate=params['learning_rate'],
                gamma=params['gamma'],
                epsilon_decay=params['epsilon_decay'],
                batch_size=params.get('batch_size', 64),
                target_update_frequency=params.get('target_update_frequency', 100)
            )
            
            # Quick training
            total_reward = 0
            
            for episode in range(n_episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                step = 0
                
                # Prepare initial data for sequences
                env_data = self.prepare_env_features(env)
                
                while not done and step < 100:  # Limit steps
                    # Get state sequence
                    state_seq = agent.prepare_sequence(env_data, env.current_step)
                    
                    # Select action
                    action = agent.select_action(state_seq)
                    
                    # Step
                    next_state, reward, done, _ = env.step(action)
                    
                    # Next sequence
                    next_state_seq = agent.prepare_sequence(env_data, env.current_step)
                    
                    # Store transition
                    agent.store_transition(state_seq, action, reward, next_state_seq, done)
                    
                    # Train
                    if len(agent.buffer) >= agent.batch_size:
                        agent.train_step()
                    
                    episode_reward += reward
                    state = next_state
                    step += 1
                
                total_reward += episode_reward
                agent.update_epsilon()
            
            # Fitness = average reward
            fitness = total_reward / n_episodes
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating particle: {e}")
            return -1000.0  # Large penalty
    
    def prepare_env_features(self, env):
        """
        Prepare features from environment data
        """
        features = ['open', 'high', 'low', 'close', 'volume',
                   'rsi', 'macd', 'macd_signal', 'macd_hist',
                   'sma_20', 'sma_50', 'ema_12', 'ema_26',
                   'bb_upper', 'bb_middle', 'bb_lower',
                   'adx', 'trend', 'price_change', 'volatility']
        
        data = env.data[features].copy()
        data = data.fillna(method='ffill').fillna(0)
        
        return data.values
    
    def optimize(self, env, input_size: int, n_final_episodes: int = 50, verbose: bool = True):
        """
        Run PSO optimization
        
        Args:
            env: Trading environment
            input_size: Number of input features
            n_final_episodes: Number of episodes for final training (default: 50)
            verbose: Print progress
        
        Returns:
            best_params, best_agent, history
        """
        # Define search space
        bounds = {
            'hidden_size': (64, 256),
            'num_layers': (1, 3),
            'sequence_length': (12, 48),
            'learning_rate': (0.0001, 0.005),
            'gamma': (0.95, 0.999),
            'epsilon_decay': (0.990, 0.999),
            'batch_size': (32, 128),
            'target_update_frequency': (50, 200)
        }
        
        # Initialize swarm
        swarm = [Particle(bounds) for _ in range(self.n_particles)]
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n=== PSO Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # Evaluate particles
            for i, particle in enumerate(swarm):
                if verbose:
                    print(f"Particle {i+1}/{self.n_particles}...", end=' ')
                
                fitness = self.evaluate_particle(particle, env, input_size)
                particle.fitness = fitness
                
                if verbose:
                    print(f"Fitness: {fitness:.2f}")
                
                # Update personal best
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = copy.deepcopy(particle.position)
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = copy.deepcopy(particle.position)
            
            self.fitness_history.append(self.global_best_fitness)
            
            if verbose:
                print(f"Global Best Fitness: {self.global_best_fitness:.2f}")
            
            # Update swarm
            for particle in swarm:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
        
        # Get best params
        best_params = {}
        for key, value in self.global_best_position.items():
            if key in ['hidden_size', 'num_layers', 'sequence_length', 
                      'batch_size', 'target_update_frequency']:
                best_params[key] = int(round(value))
            else:
                best_params[key] = value
        
        # Train final agent
        if verbose:
            print(f"\n=== Training Final Agent ===")
            print(f"Best params: {best_params}")
        
        best_agent = DQNLSTMAgent(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            sequence_length=best_params['sequence_length'],
            learning_rate=best_params['learning_rate'],
            gamma=best_params['gamma'],
            epsilon_decay=best_params['epsilon_decay'],
            batch_size=best_params['batch_size'],
            target_update_frequency=best_params['target_update_frequency']
        )
        
        # Train more thoroughly
        env_data = self.prepare_env_features(env)
        
        print(f"\nTraining final agent for {n_final_episodes} episodes...")
        print(f"Batch size: {best_agent.batch_size}, Min buffer: {min(best_agent.batch_size, 500)}\n")
        
        for episode in range(n_final_episodes):  # Configurable episodes
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done:
                state_seq = best_agent.prepare_sequence(env_data, env.current_step)
                action = best_agent.select_action(state_seq)
                next_state, reward, done, _ = env.step(action)
                next_state_seq = best_agent.prepare_sequence(env_data, env.current_step)
                
                best_agent.store_transition(state_seq, action, reward, next_state_seq, done)
                
                # Only train when buffer has enough samples
                if len(best_agent.buffer) >= min(best_agent.batch_size, 500):
                    best_agent.train_step()
                
                episode_reward += reward
                state = next_state
                steps += 1
            
            best_agent.update_epsilon()
            
            # Show progress every episode
            if verbose:
                print(f"Ep {episode + 1}/{n_final_episodes}: Reward={episode_reward:.2f}, Steps={steps}, Buffer={len(best_agent.buffer)}, Eps={best_agent.epsilon:.4f}")
        
        history = {'global_best_fitness': self.fitness_history}
        
        return best_params, best_agent, history
    
    def save_results(self, agent, params, filepath: str):
        """
        Save optimization results
        """
        import json
        from pathlib import Path
        
        # Save agent checkpoint
        agent_path = filepath.replace('.json', '_agent.pth')
        agent.save_checkpoint(agent_path, episode=50, best_reward=self.global_best_fitness)
        
        # Save params and history
        results = {
            'best_params': params,
            'global_best_fitness': float(self.global_best_fitness),
            'fitness_history': [float(x) for x in self.fitness_history],
            'agent_checkpoint': str(agent_path)
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   💾 Results saved: {filepath}")
    
    def load_results(self, filepath: str, input_size: int):
        """
        Load optimization results and agent
        """
        import json
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        best_params = results['best_params']
        
        # Create agent
        agent = DQNLSTMAgent(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            sequence_length=best_params['sequence_length'],
            learning_rate=best_params['learning_rate'],
            gamma=best_params['gamma'],
            epsilon_decay=best_params['epsilon_decay'],
            batch_size=best_params['batch_size'],
            target_update_frequency=best_params['target_update_frequency']
        )
        
        # Load checkpoint
        agent_path = results['agent_checkpoint']
        agent.load_checkpoint(agent_path)
        
        self.global_best_fitness = results['global_best_fitness']
        self.fitness_history = results['fitness_history']
        
        print(f"   ✅ Results loaded: {filepath}")
        
        return best_params, agent, {'global_best_fitness': self.fitness_history}
