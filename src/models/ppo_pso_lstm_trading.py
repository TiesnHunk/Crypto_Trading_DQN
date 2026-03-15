"""
PPO (Proximal Policy Optimization) + PSO + LSTM for Trading
Kết hợp PPO policy gradient với PSO optimization và LSTM feature extraction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
import copy


class ActorCriticLSTM(nn.Module):
    """
    Actor-Critic network với LSTM backbone cho PPO
    Actor: Output policy (action probabilities)
    Critic: Output value function (state value)
    """
    
    def __init__(self,
                 input_size: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Args:
            input_size: Số features đầu vào
            hidden_size: LSTM hidden size
            num_layers: Số LSTM layers
            dropout: Dropout rate
        """
        super(ActorCriticLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Actor head (policy)
        self.actor_fc1 = nn.Linear(hidden_size, 64)
        self.actor_fc2 = nn.Linear(64, 3)  # 3 actions
        
        # Critic head (value function)
        self.critic_fc1 = nn.Linear(hidden_size, 64)
        self.critic_fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Returns:
            action_probs: Action probabilities (batch_size, 3)
            state_value: State value (batch_size, 1)
            hidden: LSTM hidden state
        """
        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]  # Last timestep
        
        # Actor (policy)
        actor_out = self.relu(self.actor_fc1(lstm_out))
        actor_out = self.dropout(actor_out)
        action_logits = self.actor_fc2(actor_out)
        action_probs = self.softmax(action_logits)
        
        # Critic (value)
        critic_out = self.relu(self.critic_fc1(lstm_out))
        critic_out = self.dropout(critic_out)
        state_value = self.critic_fc2(critic_out)
        
        return action_probs, state_value, hidden


class PPOAgent:
    """
    PPO Agent với LSTM
    """
    
    def __init__(self,
                 input_size: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 sequence_length: int = 24,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 epsilon_clip: float = 0.2,
                 c1: float = 0.5,  # Value loss coefficient
                 c2: float = 0.01,  # Entropy coefficient
                 device: str = None):
        """
        Args:
            epsilon_clip: PPO clipping parameter
            c1: Value loss coefficient
            c2: Entropy coefficient (khuyến khích exploration)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.c1 = c1
        self.c2 = c2
        
        # Policy network
        self.policy = ActorCriticLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Memory buffer (for PPO update)
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.is_terminals = []
        
    def select_action(self, state, training=True):
        """
        Select action theo policy
        
        Args:
            state: State vector (numpy array or torch tensor)
            training: If False, choose best action (greedy)
            
        Returns:
            If training: (action, log_prob, state_value)
            If not training: action (int)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        self.policy.eval()
        
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)  # Add batch dim
            action_probs, state_value, _ = self.policy(state)
            
            if training:
                # Sample action từ distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item(), state_value.item()
            else:
                # Greedy selection
                action = torch.argmax(action_probs, dim=1)
                return action.item()
    
    def store_transition(self, state, action, reward, log_prob, state_value, is_terminal):
        """
        Store transition vào buffer
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
        self.is_terminals.append(is_terminal)
    
    def clear_memory(self):
        """
        Clear memory buffer
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.is_terminals = []
    
    def compute_returns(self):
        """
        Compute discounted returns (Monte Carlo)
        """
        returns = []
        discounted_return = 0
        
        # Reverse iteration
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        # Normalize returns
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self, epochs: int = 4):
        """
        PPO update với multiple epochs
        """
        # Compute returns
        returns = self.compute_returns()
        
        # Convert lists to tensors
        old_states = torch.stack([torch.FloatTensor(s) for s in self.states]).to(self.device)
        old_actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # PPO update (multiple epochs)
        total_loss = 0
        
        for _ in range(epochs):
            # Evaluate current policy
            self.policy.train()
            
            action_probs, state_values, _ = self.policy(old_states)
            
            # Get log probs của old actions
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            
            # Calculate advantages
            advantages = returns - state_values.squeeze()
            
            # Ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate loss
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages.detach()
            
            # PPO loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear memory
        self.clear_memory()
        
        return total_loss / epochs
    
    def predict_action(self, state: torch.Tensor) -> int:
        """
        Predict action (deterministic - chọn action có prob cao nhất)
        """
        self.policy.eval()
        
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            action_probs, _, _ = self.policy(state)
            action = torch.argmax(action_probs, dim=-1)
        
        return action.item()


class PPOPSOLSTMAgent:
    """
    Kết hợp PPO + PSO + LSTM
    - PSO: Tối ưu hyperparameters
    - LSTM: Feature extraction từ sequences
    - PPO: Policy optimization
    """
    
    def __init__(self, n_particles: int = 8, max_iterations: int = 10,
                 ppo_epochs: int = 4, w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 input_size: int = 20, device: str = None):
        self.input_size = input_size
        self.device = device
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.ppo_epochs = ppo_epochs
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.best_params = None
        self.agent = None
        self.optimizer_history = []
        
    def optimize_hyperparameters(self, env, n_particles: int = None, 
                                 max_iterations: int = None, verbose: bool = True):
        """
        Tối ưu hyperparameters bằng PSO
        
        Args:
            env: Trading environment để evaluate fitness
            n_particles: Số particles (dùng default nếu None)
            max_iterations: Số iterations (dùng default nếu None)
        """
        from .pso_lstm_trading import Particle
        
        if n_particles is None:
            n_particles = self.n_particles
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        # Define search space cho PPO hyperparameters
        bounds = {
            'hidden_size': (64, 256),
            'num_layers': (1, 3),
            'sequence_length': (12, 48),
            'learning_rate': (0.0001, 0.001),
            'epsilon_clip': (0.1, 0.3),
            'gamma': (0.95, 0.999)
        }
        
        # Initialize swarm
        swarm = [Particle(bounds) for _ in range(n_particles)]
        global_best_position = None
        global_best_fitness = -float('inf')
        fitness_history = []
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n=== PSO Iteration {iteration + 1}/{max_iterations} ===")
            
            for i, particle in enumerate(swarm):
                # Get params
                params = particle.get_integer_params()
                
                if verbose:
                    print(f"Particle {i+1}/{n_particles}: {params}")
                
                # Create PPO agent với params này
                agent = PPOAgent(
                    input_size=self.input_size,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    sequence_length=params['sequence_length'],
                    learning_rate=params['learning_rate'],
                    epsilon_clip=params['epsilon_clip'],
                    gamma=params['gamma'],
                    device=self.device
                )
                
                # Train và evaluate (simplified - chỉ vài episodes)
                fitness = self._evaluate_agent(agent, env, n_episodes=5)
                
                if verbose:
                    print(f"  Fitness: {fitness:.4f}")
                
                particle.fitness = fitness
                
                # Update personal best
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = copy.deepcopy(particle.position)
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = copy.deepcopy(particle.position)
            
            fitness_history.append(global_best_fitness)
            
            if verbose:
                print(f"Global Best Fitness: {global_best_fitness:.4f}")
            
            # Update swarm
            for particle in swarm:
                particle.update_velocity(global_best_position)
                particle.update_position()
        
        # Convert to integer params
        best_params = {}
        for key, value in global_best_position.items():
            if key in ['hidden_size', 'num_layers', 'sequence_length']:
                best_params[key] = int(round(value))
            else:
                best_params[key] = value
        
        self.best_params = best_params
        self.optimizer_history = fitness_history
        
        return best_params
    
    def optimize(self, mdp, input_size: int = 7, n_episodes: int = 50, verbose: bool = True):
        """
        Optimize và train PPO agent (wrapper method for compatibility)
        
        Returns:
            best_params, best_agent, history
        """
        # Optimize hyperparameters
        best_params = self.optimize_hyperparameters(
            mdp, 
            n_particles=self.n_particles,
            max_iterations=self.max_iterations, 
            verbose=verbose
        )
        
        # Train with best params
        rewards_history = self.train(mdp, episodes=n_episodes, verbose=verbose)
        
        history = {
            'global_best_fitness': self.optimizer_history,
            'rewards': rewards_history
        }
        
        return best_params, self.agent, history
    
    def _evaluate_agent(self, agent, env, n_episodes: int = 5):
        """
        Evaluate agent trên environment
        """
        total_reward = 0
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action
                state_tensor = torch.FloatTensor(state)
                action, log_prob, state_value = agent.select_action(state_tensor)
                
                # Step
                next_state, reward, done, _ = env.step(action)
                
                episode_reward += reward
                state = next_state
            
            total_reward += episode_reward
        
        avg_reward = total_reward / n_episodes
        return avg_reward
    
    def train(self, env, episodes: int = 100, verbose: bool = True):
        """
        Train PPO agent với best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("Chưa optimize hyperparameters! Gọi optimize_hyperparameters() trước.")
        
        # Create agent với best params
        self.agent = PPOAgent(
            input_size=self.input_size,
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            sequence_length=self.best_params['sequence_length'],
            learning_rate=self.best_params['learning_rate'],
            epsilon_clip=self.best_params['epsilon_clip'],
            gamma=self.best_params['gamma'],
            device=self.device
        )
        
        # Training loop
        rewards_history = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                state_tensor = torch.FloatTensor(state)
                action, log_prob, state_value = self.agent.select_action(state_tensor)
                
                # Step
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                self.agent.store_transition(state, action, reward, log_prob, state_value, done)
                
                episode_reward += reward
                state = next_state
            
            # Update policy
            loss = self.agent.update()
            
            rewards_history.append(episode_reward)
            
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Episode {episode + 1}/{episodes}: Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
        
        return rewards_history
    
    def predict_action(self, state: np.ndarray) -> int:
        """
        Predict action
        """
        if self.agent is None:
            raise ValueError("Model chưa được train!")
        
        state_tensor = torch.FloatTensor(state)
        return self.agent.predict_action(state_tensor)


# Alias for compatibility
PPOPSOOptimizer = PPOPSOLSTMAgent
