"""
LSTM Model for Trading Action Prediction
Dự báo hành động Buy/Sell/Hold sử dụng LSTM
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class LSTMTradingModel(nn.Module):
    """
    LSTM model để dự báo hành động trading
    Input: Sequence of price + indicators
    Output: Action probabilities (Buy/Sell/Hold)
    """
    
    def __init__(self, 
                 input_size: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 output_size: int = 3,
                 dropout: float = 0.2):
        """
        Args:
            input_size: Số features đầu vào (price + indicators)
            hidden_size: Số units trong LSTM layer
            num_layers: Số LSTM layers
            output_size: 3 actions (Buy=0, Sell=1, Hold=2)
            dropout: Dropout rate để tránh overfitting
        """
        super(LSTMTradingModel, self).__init__()
        
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
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            hidden: Hidden state (optional)
            
        Returns:
            Action probabilities (batch_size, 3)
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Lấy output của timestep cuối cùng
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        # Softmax để có probability distribution
        out = self.softmax(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        """
        Khởi tạo hidden state
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden


class LSTMTradingAgent:
    """
    Agent sử dụng LSTM để trading
    """
    
    def __init__(self,
                 input_size: int = 20,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 sequence_length: int = 24,
                 learning_rate: float = 0.001,
                 device: str = None):
        """
        Args:
            input_size: Số features
            hidden_size: LSTM hidden size
            num_layers: Số LSTM layers
            sequence_length: Độ dài sequence để predict
            learning_rate: Learning rate
            device: 'cuda' hoặc 'cpu'
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.sequence_length = sequence_length
        self.input_size = input_size
        
        # Model
        self.model = LSTMTradingModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        
        # Optimizer và loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Scaler để normalize data
        self.scaler = MinMaxScaler()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def prepare_sequences(self, data: np.ndarray, labels: np.ndarray = None) -> Tuple:
        """
        Chuẩn bị sequences từ data
        
        Args:
            data: Array (n_samples, n_features)
            labels: Array (n_samples,) - actions
            
        Returns:
            X, y tensors
        """
        n_samples = len(data) - self.sequence_length
        
        X = []
        y = []
        
        for i in range(n_samples):
            # Lấy sequence
            seq = data[i:i+self.sequence_length]
            X.append(seq)
            
            if labels is not None:
                # Label cho timestep tiếp theo
                y.append(labels[i + self.sequence_length])
        
        X = np.array(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if labels is not None:
            y_tensor = torch.LongTensor(y).to(self.device)
            return X_tensor, y_tensor
        
        return X_tensor, None
    
    def train_epoch(self, X_train, y_train, batch_size: int = 32):
        """
        Train 1 epoch
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            # Get batch
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(batch_X)
            
            # Calculate loss
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, X_val, y_val):
        """
        Validate model
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs, _ = self.model(X_val)
            loss = self.criterion(outputs, y_val)
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_val).float().mean()
        
        return loss.item(), accuracy.item()
    
    def train(self, X_train, y_train, 
              epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, patience: int = 10, verbose: bool = True):
        """
        Train model với early stopping
        """
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        
        # Split validation
        val_size = int(len(X_train_seq) * validation_split)
        train_size = len(X_train_seq) - val_size
        
        X_train_final = X_train_seq[:train_size]
        y_train_final = y_train_seq[:train_size]
        X_val = X_train_seq[train_size:]
        y_val = y_train_seq[train_size:]
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train_final, y_train_final, batch_size)
            
            # Validate
            val_loss, val_acc = self.validate(X_val, y_val)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                # Load best model
                self.model.load_state_dict(self.best_model_state)
                break
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
    
    def predict(self, X_test):
        """
        Predict actions cho test set
        
        Args:
            X_test: Test features (n_samples, n_features)
            
        Returns:
            predictions: Array of actions (n_samples - sequence_length,)
        """
        self.model.eval()
        
        # Prepare sequences
        X_test_seq, _ = self.prepare_sequences(X_test)
        
        predictions = []
        
        with torch.no_grad():
            for i in range(len(X_test_seq)):
                outputs, _ = self.model(X_test_seq[i:i+1])
                action = torch.argmax(outputs, dim=1).item()
                predictions.append(action)
        
        return np.array(predictions)
    
    def predict_action(self, sequence: np.ndarray) -> int:
        """
        Predict action cho 1 sequence
        
        Args:
            sequence: Array (sequence_length, n_features)
            
        Returns:
            action: 0=Buy, 1=Sell, 2=Hold
        """
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensor
            X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            outputs, _ = self.model(X)
            action = torch.argmax(outputs, dim=1).item()
        
        return action
    
    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Predict probabilities cho mỗi action
        
        Returns:
            proba: Array (3,) - probabilities for Buy/Sell/Hold
        """
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            outputs, _ = self.model(X)
            proba = outputs.cpu().numpy()[0]
        
        return proba
    
    def save(self, path: str):
        """
        Save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'sequence_length': self.sequence_length,
            'input_size': self.input_size
        }, path)
    
    def load(self, path: str):
        """
        Load model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
