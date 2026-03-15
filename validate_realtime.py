"""
Real-time Validation Script
Tracks DQN predictions on live data and validates accuracy
"""

import torch
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.dqn_agent import DQNAgent
from data.binance_data import BinanceDataFetcher
from utils.indicators import TechnicalIndicators


class RealtimeValidator:
    def __init__(self, checkpoint_path, predictions_file="realtime_predictions.json"):
        self.checkpoint_path = Path(checkpoint_path)
        self.predictions_file = Path(predictions_file)
        self.agent = None
        self.fetcher = None
        self.predictions = []
        
        # Load existing predictions if file exists
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                self.predictions = json.load(f)
            print(f"📂 Loaded {len(self.predictions)} existing predictions")
    
    def load_agent(self):
        """Load DQN agent"""
        print(f"📦 Loading DQN agent from {self.checkpoint_path.name}...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        hyperparams = checkpoint.get('hyperparameters', {})
        state_dim = hyperparams.get('state_dim', 7)
        action_dim = hyperparams.get('action_dim', 3)
        
        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device='cpu'
        )
        
        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.agent.epsilon = 0.01
        
        print("✅ Agent loaded")
        return True
    
    def initialize_fetcher(self):
        """Initialize Binance data fetcher"""
        try:
            self.fetcher = BinanceDataFetcher()
            print("✅ Binance fetcher initialized")
            return True
        except Exception as e:
            print(f"❌ Error initializing fetcher: {e}")
            return False
    
    def get_current_data(self, symbol='BTCUSDT', lookback_hours=100):
        """Fetch current data from Binance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=lookback_hours)
        
        df = self.fetcher.get_klines(
            symbol=symbol,
            interval='1h',
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        if df is None or len(df) == 0:
            return None
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            if 'open_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            elif df.index.name in ['timestamp', 'open_time', 'time']:
                # Reset index to make it a column
                df = df.reset_index()
                if 'index' in df.columns:
                    df.rename(columns={'index': 'timestamp'}, inplace=True)
            else:
                # Create timestamp from current time backwards
                df['timestamp'] = pd.date_range(
                    end=datetime.now(), 
                    periods=len(df), 
                    freq='1H'
                )
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add technical indicators
        df = TechnicalIndicators.add_all_indicators(df)
        
        return df
    
    def extract_state(self, df, index=-1):
        """Extract DQN state from data (8 features matching training)"""
        row = df.iloc[index]
        
        # Feature 1: Position (0 = no position)
        position = 0.0
        
        # Feature 2: RSI normalized [0, 1]
        rsi_norm = row.get('rsi', 50) / 100.0 if 'rsi' in row and not pd.isna(row['rsi']) else 0.5
        
        # Feature 3: MACD histogram normalized [-1, 1]
        macd_hist = row.get('macd_histogram', 0)
        macd_norm = np.clip(macd_hist / 100.0, -1, 1) if 'macd_histogram' in row and not pd.isna(macd_hist) else 0.0
        
        # Feature 4: Trend (EMA20 vs EMA50) normalized [0, 1]
        if 'ema_20' in row and 'ema_50' in row and not pd.isna(row['ema_20']) and not pd.isna(row['ema_50']):
            trend_signal = (row['ema_20'] - row['ema_50']) / row['ema_50']
            trend_norm = np.clip((trend_signal + 0.1) / 0.2, 0, 1)
        else:
            trend_norm = 0.5
        
        # Feature 5: Bollinger Bands position [0, 1]
        if all(k in row for k in ['close', 'bb_upper', 'bb_lower']):
            bb_range = row['bb_upper'] - row['bb_lower']
            bb_position = (row['close'] - row['bb_lower']) / bb_range if bb_range > 0 else 0.5
            bb_position = np.clip(bb_position, 0, 1)
        else:
            bb_position = 0.5
        
        # Feature 6: Volatility (ATR %) normalized [0, 1]
        if 'atr' in row and not pd.isna(row['atr']):
            volatility_pct = row['atr'] / row['close'] if row['close'] > 0 else 0
            volatility_norm = min(volatility_pct / 0.1, 1.0)
        else:
            volatility_norm = 0.5
        
        # Feature 7: Current profit (0.5 for no position)
        current_profit_norm = 0.5
        
        # Feature 8: Time since trade (0.0 for no active trade)
        time_since_trade_norm = 0.0
        
        return np.array([
            position, rsi_norm, macd_norm, trend_norm,
            bb_position, volatility_norm, current_profit_norm, time_since_trade_norm
        ], dtype=np.float32)
    
    def make_prediction(self, symbol='BTCUSDT', validation_hours=1):
        """Make a prediction and save it for later validation"""
        print(f"\n🔮 Making prediction for {symbol}...")
        
        # Get current data
        df = self.get_current_data(symbol)
        if df is None:
            print("❌ Failed to fetch data")
            return None
        
        # Extract state
        state = self.extract_state(df)
        current_row = df.iloc[-1]
        current_price = current_row['close']
        current_time = current_row['timestamp']
        
        # Get prediction
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent.policy_net(state_tensor).squeeze(0).numpy()
        
        action = int(np.argmax(q_values))
        action_names = ['BUY', 'SELL', 'HOLD']
        
        # Calculate confidence
        exp_q = np.exp(q_values - np.max(q_values))
        probabilities = exp_q / np.sum(exp_q)
        confidence = probabilities[action]
        
        # Save prediction
        prediction = {
            'id': len(self.predictions) + 1,
            'timestamp': current_time.isoformat(),
            'symbol': symbol,
            'price': float(current_price),
            'action': action,
            'action_name': action_names[action],
            'confidence': float(confidence),
            'q_values': {
                'buy': float(q_values[0]),
                'sell': float(q_values[1]),
                'hold': float(q_values[2])
            },
            'indicators': {
                'rsi': float(current_row.get('rsi', 0)),
                'macd': float(current_row.get('macd_histogram', 0)),
                'adx': float(current_row.get('adx', 0))
            },
            'validation_time': (current_time + timedelta(hours=validation_hours)).isoformat(),
            'validated': False,
            'actual_price': None,
            'price_change_pct': None,
            'correct': None
        }
        
        self.predictions.append(prediction)
        self._save_predictions()
        
        print(f"✅ Prediction saved:")
        print(f"   Price: ${current_price:,.2f}")
        print(f"   Action: {action_names[action]} ({confidence*100:.1f}% confidence)")
        print(f"   Will validate at: {prediction['validation_time']}")
        
        return prediction
    
    def validate_predictions(self, symbol='BTCUSDT'):
        """Validate past predictions that are due"""
        print(f"\n🔍 Validating predictions for {symbol}...")
        
        current_time = datetime.now()
        validated_count = 0
        
        for pred in self.predictions:
            # Skip already validated or wrong symbol
            if pred['validated'] or pred['symbol'] != symbol:
                continue
            
            # Check if validation time has passed
            validation_time = datetime.fromisoformat(pred['validation_time'])
            pred_time = datetime.fromisoformat(pred['timestamp'])
            
            # Must wait at least 1 hour from prediction time
            time_since_prediction = (current_time - pred_time).total_seconds() / 3600
            if time_since_prediction < 1.0:
                print(f"⏳ Prediction #{pred['id']}: Only {time_since_prediction:.1f}h passed, need 1.0h")
                continue
            
            # Must be past validation time
            if current_time < validation_time:
                time_until_validation = (validation_time - current_time).total_seconds() / 60
                print(f"⏳ Prediction #{pred['id']}: {time_until_validation:.0f}m until validation")
                continue
            
            # Fetch fresh data to get current price
            df = self.get_current_data(symbol, lookback_hours=24)
            if df is None or len(df) == 0:
                print(f"❌ Prediction #{pred['id']}: Failed to fetch data")
                continue
            
            # Use the most recent price (current market price)
            actual_price = df.iloc[-1]['close']
            
            # Calculate price change
            original_price = pred['price']
            price_change_pct = ((actual_price - original_price) / original_price) * 100
            
            # Determine if prediction was correct
            correct = self._is_prediction_correct(pred['action'], price_change_pct)
            
            # Update prediction
            pred['validated'] = True
            pred['actual_price'] = float(actual_price)
            pred['price_change_pct'] = float(price_change_pct)
            pred['correct'] = correct
            
            validated_count += 1
            
            print(f"\n📊 Validation #{pred['id']}:")
            print(f"   Predicted: {pred['action_name']}")
            print(f"   Original Price: ${original_price:,.2f}")
            print(f"   Actual Price: ${actual_price:,.2f}")
            print(f"   Change: {price_change_pct:+.2f}%")
            print(f"   Result: {'✅ CORRECT' if correct else '❌ WRONG'}")
        
        if validated_count > 0:
            self._save_predictions()
            print(f"\n✅ Validated {validated_count} predictions")
            self.print_statistics()
        else:
            print("ℹ️  No predictions due for validation")
    
    def _is_prediction_correct(self, action, price_change_pct, threshold=1.0):
        """Check if prediction was correct based on price change"""
        # BUY (0): Correct if price increased > threshold
        if action == 0:
            return price_change_pct > threshold
        # SELL (1): Correct if price decreased < -threshold
        elif action == 1:
            return price_change_pct < -threshold
        # HOLD (2): Correct if price stayed within [-threshold, threshold]
        else:
            return abs(price_change_pct) <= threshold
    
    def _save_predictions(self):
        """Save predictions to file (convert numpy types to Python native)"""
        # Convert numpy types to native Python types
        predictions_clean = []
        for pred in self.predictions:
            pred_clean = {}
            for key, value in pred.items():
                if isinstance(value, (np.integer, np.floating)):
                    pred_clean[key] = float(value)
                elif isinstance(value, np.bool_):
                    pred_clean[key] = bool(value)
                elif isinstance(value, dict):
                    # Handle nested dicts (like q_values, indicators)
                    pred_clean[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                      for k, v in value.items()}
                else:
                    pred_clean[key] = value
            predictions_clean.append(pred_clean)
        
        with open(self.predictions_file, 'w') as f:
            json.dump(predictions_clean, f, indent=2)
    
    def print_statistics(self):
        """Print validation statistics"""
        validated = [p for p in self.predictions if p['validated']]
        
        if len(validated) == 0:
            print("\nℹ️  No validated predictions yet")
            return
        
        correct = sum(1 for p in validated if p['correct'])
        total = len(validated)
        accuracy = (correct / total) * 100
        
        # By action
        by_action = {'BUY': [], 'SELL': [], 'HOLD': []}
        for p in validated:
            by_action[p['action_name']].append(p['correct'])
        
        print("\n" + "="*70)
        print("📊 VALIDATION STATISTICS")
        print("="*70)
        print(f"Total Predictions: {len(self.predictions)}")
        print(f"Validated: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        print("\n By Action:")
        for action, results in by_action.items():
            if len(results) > 0:
                action_accuracy = (sum(results) / len(results)) * 100
                print(f"   {action}: {action_accuracy:.2f}% ({sum(results)}/{len(results)})")
        
        # Average profit
        avg_change = np.mean([p['price_change_pct'] for p in validated])
        print(f"\nAverage Price Change: {avg_change:+.2f}%")
        
        print("="*70)
    
    def run_continuous(self, symbol='BTCUSDT', predict_interval=3600, validate_interval=600):
        """Run continuous prediction and validation"""
        print("\n🚀 Starting continuous real-time validation...")
        print(f"   Symbol: {symbol}")
        print(f"   Predict every: {predict_interval}s ({predict_interval/3600:.1f}h)")
        print(f"   Validate every: {validate_interval}s ({validate_interval/60:.1f}m)")
        print("\nPress Ctrl+C to stop\n")
        
        last_prediction = 0
        last_validation = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Make prediction
                if current_time - last_prediction >= predict_interval:
                    self.make_prediction(symbol)
                    last_prediction = current_time
                
                # Validate predictions
                if current_time - last_validation >= validate_interval:
                    self.validate_predictions(symbol)
                    last_validation = current_time
                
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            print("\n\n⏸️  Stopped by user")
            self.print_statistics()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time DQN Validation")
    parser.add_argument('--checkpoint', type=str, 
                       default='src/checkpoints_dqn/checkpoint_latest.pkl',
                       help='Path to DQN checkpoint')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading symbol (e.g., BTCUSDT, ETHUSDT)')
    parser.add_argument('--mode', type=str, choices=['predict', 'validate', 'continuous', 'stats'],
                       default='continuous',
                       help='Operation mode')
    parser.add_argument('--predict-interval', type=int, default=3600,
                       help='Prediction interval in seconds (default: 1 hour)')
    parser.add_argument('--validate-interval', type=int, default=600,
                       help='Validation check interval in seconds (default: 10 min)')
    
    args = parser.parse_args()
    
    validator = RealtimeValidator(args.checkpoint)
    
    # Load agent and fetcher
    if not validator.load_agent():
        sys.exit(1)
    if not validator.initialize_fetcher():
        sys.exit(1)
    
    # Run based on mode
    if args.mode == 'predict':
        validator.make_prediction(args.symbol)
    elif args.mode == 'validate':
        validator.validate_predictions(args.symbol)
    elif args.mode == 'stats':
        validator.print_statistics()
    elif args.mode == 'continuous':
        validator.run_continuous(
            args.symbol,
            args.predict_interval,
            args.validate_interval
        )
