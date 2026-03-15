"""
PSO (Particle Swarm Optimization) + LSTM for Trading
Tối ưu hóa hyperparameters của LSTM bằng PSO
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import copy
from .lstm_trading import LSTMTradingAgent


class Particle:
    """
    Một hạt (particle) trong PSO swarm
    Mỗi particle đại diện cho 1 bộ hyperparameters
    """
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        """
        Args:
            bounds: Dictionary {param_name: (min, max)}
        """
        self.bounds = bounds
        self.position = {}
        self.velocity = {}
        self.best_position = {}
        self.best_fitness = -float('inf')
        self.fitness = -float('inf')
        
        # Khởi tạo position và velocity ngẫu nhiên
        for param, (min_val, max_val) in bounds.items():
            # Position: random trong bounds
            self.position[param] = np.random.uniform(min_val, max_val)
            
            # Velocity: random trong [-range/10, range/10]
            range_val = max_val - min_val
            self.velocity[param] = np.random.uniform(-range_val/10, range_val/10)
        
        self.best_position = copy.deepcopy(self.position)
    
    def update_velocity(self, global_best_position: Dict, w: float = 0.7, 
                        c1: float = 1.5, c2: float = 1.5):
        """
        Cập nhật velocity theo công thức PSO
        
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        
        Args:
            global_best_position: Best position của toàn swarm
            w: Inertia weight (quán tính)
            c1: Cognitive coefficient (hướng đến best cá nhân)
            c2: Social coefficient (hướng đến best toàn cục)
        """
        for param in self.position.keys():
            r1 = np.random.random()
            r2 = np.random.random()
            
            cognitive = c1 * r1 * (self.best_position[param] - self.position[param])
            social = c2 * r2 * (global_best_position[param] - self.position[param])
            
            self.velocity[param] = (w * self.velocity[param] + 
                                   cognitive + social)
    
    def update_position(self):
        """
        Cập nhật position dựa vào velocity
        """
        for param in self.position.keys():
            # Update position
            self.position[param] += self.velocity[param]
            
            # Clamp vào bounds
            min_val, max_val = self.bounds[param]
            self.position[param] = np.clip(self.position[param], min_val, max_val)
    
    def get_integer_params(self) -> Dict:
        """
        Convert continuous position thành integer params cho LSTM
        """
        params = {}
        for key, value in self.position.items():
            if key in ['hidden_size', 'num_layers', 'sequence_length']:
                params[key] = int(round(value))
            else:
                params[key] = value
        return params


class PSOLSTMOptimizer:
    """
    Tối ưu hyperparameters của LSTM bằng PSO
    """
    
    def __init__(self, 
                 n_particles: int = 10,
                 max_iterations: int = 20,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        Args:
            n_particles: Số particles trong swarm
            max_iterations: Số vòng lặp tối đa
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
        """
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Define search space cho LSTM hyperparameters
        self.bounds = {
            'hidden_size': (32, 256),      # LSTM hidden units
            'num_layers': (1, 4),           # Số LSTM layers
            'sequence_length': (12, 48),    # Độ dài sequence (12-48 hours)
            'learning_rate': (0.0001, 0.01), # Learning rate
            'dropout': (0.1, 0.5)            # Dropout rate
        }
        
        # Initialize swarm
        self.swarm = [Particle(self.bounds) for _ in range(n_particles)]
        
        # Global best
        self.global_best_position = None
        self.global_best_fitness = -float('inf')
        
        # History
        self.fitness_history = []
        
    def evaluate_particle(self, particle: Particle, 
                         X_train, y_train, X_val, y_val,
                         input_size: int) -> float:
        """
        Đánh giá fitness của 1 particle (train LSTM và measure performance)
        
        Args:
            particle: Particle cần đánh giá
            X_train, y_train: Training data
            X_val, y_val: Validation data
            input_size: Số features đầu vào
            
        Returns:
            fitness: Validation accuracy (càng cao càng tốt)
        """
        params = particle.get_integer_params()
        
        try:
            # Tạo LSTM agent với params từ particle
            agent = LSTMTradingAgent(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                sequence_length=params['sequence_length'],
                learning_rate=params['learning_rate']
            )
            
            # Prepare sequences với sequence_length từ particle
            X_train_seq, y_train_seq = agent.prepare_sequences(
                X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train,
                y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
            )
            
            X_val_seq, y_val_seq = agent.prepare_sequences(
                X_val.cpu().numpy() if isinstance(X_val, torch.Tensor) else X_val,
                y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val
            )
            
            # Train model (epochs ngắn để PSO nhanh hơn)
            agent.train(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                epochs=30,  # Giảm epochs để PSO nhanh
                batch_size=32,
                patience=5,
                verbose=False
            )
            
            # Evaluate
            val_loss, val_acc = agent.validate(X_val_seq, y_val_seq)
            
            # Fitness = validation accuracy
            fitness = val_acc
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating particle: {e}")
            return -1.0  # Penalty cho invalid params
    
    def optimize(self, X_train, y_train, input_size: int,
                validation_split: float = 0.2,
                epochs_per_eval: int = 20,
                verbose: bool = True):
        """
        Chạy PSO optimization
        
        Args:
            X_train, y_train: Training data (raw, chưa prepare sequences)
            input_size: Số features
            validation_split: Train/val split ratio
            epochs_per_eval: Epochs để train mỗi particle
            verbose: In progress
            
        Returns:
            best_params: Dict chứa best hyperparameters
            best_agent: LSTM agent với best params đã train
            history: Training history
        """
        # Split validation
        val_size = int(len(X_train) * validation_split)
        train_size = len(X_train) - val_size
        
        X_train_split = X_train[:train_size]
        y_train_split = y_train[:train_size]
        X_val = X_train[train_size:]
        y_val = y_train[train_size:]
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n=== PSO Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # Evaluate tất cả particles
            for i, particle in enumerate(self.swarm):
                if verbose:
                    print(f"Evaluating particle {i+1}/{self.n_particles}...", end=' ')
                
                # Evaluate fitness
                fitness = self.evaluate_particle(
                    particle, X_train_split, y_train_split, X_val, y_val, input_size
                )
                
                particle.fitness = fitness
                
                if verbose:
                    print(f"Fitness: {fitness:.4f}")
                
                # Update personal best
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = copy.deepcopy(particle.position)
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = copy.deepcopy(particle.position)
            
            # Record iteration best
            self.fitness_history.append(self.global_best_fitness)
            
            if verbose:
                print(f"Global Best Fitness: {self.global_best_fitness:.4f}")
            
            # Update velocities và positions
            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, 
                                        self.w, self.c1, self.c2)
                particle.update_position()
        
        # Convert best position to integer params
        best_params = {}
        for key, value in self.global_best_position.items():
            if key in ['hidden_size', 'num_layers', 'sequence_length']:
                best_params[key] = int(round(value))
            else:
                best_params[key] = value
        
        # Train final agent with best params
        if verbose:
            print(f"\n=== Training Final Agent with Best Params ===")
            print(f"Best params: {best_params}")
        
        best_agent = LSTMTradingAgent(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            sequence_length=best_params['sequence_length'],
            learning_rate=best_params['learning_rate']
        )
        
        # Train với nhiều epochs hơn
        history = best_agent.train(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_split=validation_split,
            patience=10,
            verbose=verbose
        )
        
        return best_params, best_agent, {'global_best_fitness': self.fitness_history}
        best_params = {}
        for key, value in self.global_best_position.items():
            if key in ['hidden_size', 'num_layers', 'sequence_length']:
                best_params[key] = int(round(value))
            else:
                best_params[key] = value
        
        return best_params


class PSOLSTMAgent:
    """
    LSTM Agent với hyperparameters được tối ưu bằng PSO
    """
    
    def __init__(self, input_size: int = 20, device: str = None):
        """
        Args:
            input_size: Số features
            device: 'cuda' hoặc 'cpu'
        """
        self.input_size = input_size
        self.device = device
        self.best_params = None
        self.agent = None
        self.optimizer_history = []
        
    def optimize_and_train(self, X_train, y_train, X_val, y_val,
                          n_particles: int = 10,
                          max_iterations: int = 20,
                          final_epochs: int = 100,
                          verbose: bool = True):
        """
        Tối ưu hyperparameters bằng PSO và train model cuối cùng
        
        Args:
            X_train, y_train: Training data (numpy arrays)
            X_val, y_val: Validation data
            n_particles: Số particles trong PSO
            max_iterations: Số iterations của PSO
            final_epochs: Số epochs train model cuối cùng
            verbose: Print progress
        """
        # 1. PSO Optimization
        if verbose:
            print("=" * 60)
            print("PHASE 1: PSO Hyperparameter Optimization")
            print("=" * 60)
        
        optimizer = PSOLSTMOptimizer(
            n_particles=n_particles,
            max_iterations=max_iterations
        )
        
        self.best_params = optimizer.optimize(
            X_train, y_train, X_val, y_val, 
            self.input_size, verbose=verbose
        )
        
        self.optimizer_history = optimizer.fitness_history
        
        if verbose:
            print("\n" + "=" * 60)
            print("PSO Optimization Complete!")
            print(f"Best Hyperparameters: {self.best_params}")
            print(f"Best Fitness: {optimizer.global_best_fitness:.4f}")
            print("=" * 60)
        
        # 2. Train final model với best params
        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 2: Training Final Model with Best Hyperparameters")
            print("=" * 60)
        
        self.agent = LSTMTradingAgent(
            input_size=self.input_size,
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            sequence_length=self.best_params['sequence_length'],
            learning_rate=self.best_params['learning_rate'],
            device=self.device
        )
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.agent.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.agent.prepare_sequences(X_val, y_val)
        
        # Train với nhiều epochs hơn
        self.agent.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            epochs=final_epochs,
            batch_size=32,
            patience=15,
            verbose=verbose
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
    
    def predict_action(self, sequence: np.ndarray) -> int:
        """
        Predict action
        """
        if self.agent is None:
            raise ValueError("Model chưa được train! Gọi optimize_and_train() trước.")
        return self.agent.predict_action(sequence)
    
    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        """
        if self.agent is None:
            raise ValueError("Model chưa được train! Gọi optimize_and_train() trước.")
        return self.agent.predict_proba(sequence)
    
    def save(self, path: str):
        """
        Save model
        """
        if self.agent is None:
            raise ValueError("Model chưa được train!")
        
        # Save agent
        self.agent.save(path)
        
        # Save best params
        import json
        params_path = path.replace('.pth', '_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
    
    def load(self, path: str):
        """
        Load model
        """
        # Load params
        import json
        params_path = path.replace('.pth', '_params.json')
        with open(params_path, 'r') as f:
            self.best_params = json.load(f)
        
        # Create agent
        self.agent = LSTMTradingAgent(
            input_size=self.input_size,
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            sequence_length=self.best_params['sequence_length'],
            learning_rate=self.best_params['learning_rate'],
            device=self.device
        )
        
        # Load weights
        self.agent.load(path)
