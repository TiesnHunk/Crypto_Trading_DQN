"""
Flask Web Application - Bitcoin Trading Predictor
Ứng dụng Web dự đoán giao dịch Bitcoin theo thời gian thực
Updated to use DQN Model (V6.4)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Import project modules
from src.data.binance_data import BinanceDataFetcher
from src.utils.indicators import TechnicalIndicators
from src.models.dqn_agent import DQNAgent
from src.models.mdp_trading import TradingMDP

app = Flask(__name__)
CORS(app)

# Global variables
dqn_agent = None
fetcher = None
# Use DQN best checkpoint (Episode 1,831)
MODEL_PATH = os.path.join(project_root, 'src', 'checkpoints_dqn', 'checkpoint_best.pkl')

def load_model():
    """Load trained DQN model"""
    global dqn_agent
    try:
        model_path = os.path.join(project_root, 'src', 'checkpoints_dqn', 'checkpoint_best.pkl')
        
        if not os.path.exists(model_path):
            print(f"⚠️ DQN model not found at {model_path}")
            return False
        
        print(f"📦 Loading DQN model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        episode = checkpoint.get('episode', 0)
        epsilon = checkpoint.get('epsilon', 0.01)
        
        print(f"   Episode: {episode}")
        print(f"   Epsilon: {epsilon:.6f}")
        
        # Create DQN agent
        hyperparams = checkpoint.get('hyperparameters', {})
        state_dim = hyperparams.get('state_dim', 7)
        action_dim = hyperparams.get('action_dim', 3)
        
        dqn_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device='cpu'
        )
        
        # Load weights
        dqn_agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        dqn_agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        dqn_agent.epsilon = 0.01  # Use minimal exploration for production
        
        print("✅ DQN model loaded successfully")
        print(f"   State dim: {state_dim}, Action dim: {action_dim}")
        print(f"   Parameters: {sum(p.numel() for p in dqn_agent.policy_net.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading DQN model: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_fetcher():
    """Initialize Binance data fetcher"""
    global fetcher
    try:
        fetcher = BinanceDataFetcher()
        print("✅ Binance fetcher initialized")
        return True
    except Exception as e:
        print(f"❌ Error initializing fetcher: {e}")
        return False

def get_realtime_data(symbol='BTCUSDT', interval='1h', limit=100):
    """Fetch real-time data from Binance"""
    try:
        # Get data from last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        df = fetcher.get_klines(
            symbol=symbol, 
            interval=interval, 
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        if df is None or len(df) == 0:
            return None
        
        # Add technical indicators
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Get latest data
        df = df.tail(limit)
        
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def get_state(df, index):
    """Extract DQN state (7 features) from dataframe"""
    try:
        row = df.iloc[index]
        
        # DQN State: [position, cash, holdings, rsi, macd_histogram, bb_position, adx]
        
        # Position (0 = no position, 1 = holding) - always 0 for new prediction
        position = 0.0
        
        # RSI normalized [0, 1]
        if 'rsi' in row and not pd.isna(row['rsi']):
            rsi_norm = row['rsi'] / 100.0
        else:
            rsi_norm = 0.5
        
        # MACD histogram normalized [-1, 1]
        if 'macd_histogram' in row and not pd.isna(row['macd_histogram']):
            macd_hist = row['macd_histogram']
            macd_norm = np.clip(macd_hist / 100.0, -1, 1)
        else:
            macd_norm = 0.0
        
        # Trend normalized [0, 1] - EMA20 vs EMA50
        if 'ema_20' in row and 'ema_50' in row and not pd.isna(row['ema_20']) and not pd.isna(row['ema_50']):
            trend_signal = (row['ema_20'] - row['ema_50']) / row['ema_50']
            trend_norm = np.clip((trend_signal + 0.1) / 0.2, 0, 1)  # -10% to +10% -> 0 to 1
        else:
            trend_norm = 0.5
        
        # Bollinger Bands position [0, 1]
        if all(k in row for k in ['close', 'bb_upper', 'bb_lower']):
            bb_range = row['bb_upper'] - row['bb_lower']
            if bb_range > 0:
                bb_position = (row['close'] - row['bb_lower']) / bb_range
                bb_position = np.clip(bb_position, 0, 1)
            else:
                bb_position = 0.5
        else:
            bb_position = 0.5
        
        # Volatility normalized [0, 1] - ATR percentage
        if 'atr' in row and not pd.isna(row['atr']):
            volatility_pct = row['atr'] / row['close'] if row['close'] > 0 else 0
            volatility_norm = min(volatility_pct / 0.1, 1.0)  # 10% max ATR
        else:
            volatility_norm = 0.5
        
        # Current profit (always 0.5 for first prediction - no position held)
        current_profit_norm = 0.5
        
        # Time since trade (always 0.0 for first prediction - no active trade)
        time_since_trade_norm = 0.0
        
        features = np.array([
            position,
            rsi_norm,
            macd_norm,
            trend_norm,
            bb_position,
            volatility_norm,
            current_profit_norm,
            time_since_trade_norm
        ], dtype=np.float32)
        
        return features
    except Exception as e:
        print(f"❌ Error extracting state: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_action(state):
    """Predict trading action using DQN model"""
    try:
        if dqn_agent is None:
            print("❌ DQN agent not loaded")
            return None
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get Q-values from DQN
        with torch.no_grad():
            q_values = dqn_agent.policy_net(state_tensor).squeeze(0).numpy()
        
        # Choose action with highest Q-value (greedy for production)
        action = int(np.argmax(q_values))
        
        # Calculate confidence (softmax probability)
        exp_q = np.exp(q_values - np.max(q_values))
        probabilities = exp_q / np.sum(exp_q)
        
        action_names = ['MUA', 'BÁN', 'GIỮ']
        confidence = probabilities[action] * 100
        
        return {
            'action': action,
            'action_name': action_names[action],
            'confidence': float(confidence),
            'probabilities': {
                'buy': float(probabilities[0] * 100),
                'sell': float(probabilities[1] * 100),
                'hold': float(probabilities[2] * 100)
            },
            'q_values': {
                'buy': float(q_values[0]),
                'sell': float(q_values[1]),
                'hold': float(q_values[2])
            }
        }
    except Exception as e:
        print(f"❌ Error predicting action: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_recommendation(prediction, df):
    """Generate trading recommendation with detailed analysis"""
    try:
        latest = df.iloc[-1]
        action_name = prediction['action_name']
        confidence = prediction['confidence']
        
        # Determine risk level
        if confidence >= 70:
            risk_level = "THẤP"
            risk_color = "success"
        elif confidence >= 50:
            risk_level = "TRUNG BÌNH"
            risk_color = "warning"
        else:
            risk_level = "CAO"
            risk_color = "danger"
        
        # Generate recommendation text
        recommendations = {
            'MUA': {
                'title': '📈 KHUYẾN NGHỊ MUA',
                'message': f'Model dự đoán xu hướng tăng giá với độ tin cậy {confidence:.1f}%',
                'details': [
                    f'RSI: {latest["rsi"]:.1f} - {"Oversold" if latest["rsi"] < 30 else "Neutral" if latest["rsi"] < 70 else "Overbought"}',
                    f'MACD: {latest["macd"]:.2f} (Signal: {latest["macd_signal"]:.2f})',
                    f'Bollinger: Giá ở {"dưới" if latest["close"] < latest["bb_middle"] else "trên"} đường giữa',
                    f'Khối lượng: {latest["volume"]:,.0f}'
                ],
                'advice': 'Đây có thể là thời điểm tốt để mua vào. Tuy nhiên, hãy cân nhắc thêm các yếu tố thị trường khác.'
            },
            'BÁN': {
                'title': '📉 KHUYẾN NGHỊ BÁN',
                'message': f'Model dự đoán xu hướng giảm giá với độ tin cậy {confidence:.1f}%',
                'details': [
                    f'RSI: {latest["rsi"]:.1f} - {"Oversold" if latest["rsi"] < 30 else "Neutral" if latest["rsi"] < 70 else "Overbought"}',
                    f'MACD: {latest["macd"]:.2f} (Signal: {latest["macd_signal"]:.2f})',
                    f'Bollinger: Giá ở {"dưới" if latest["close"] < latest["bb_middle"] else "trên"} đường giữa',
                    f'Khối lượng: {latest["volume"]:,.0f}'
                ],
                'advice': 'Model dự đoán giá có thể giảm. Cân nhắc bán để cắt lỗ hoặc chốt lời.'
            },
            'GIỮ': {
                'title': '⏸️ KHUYẾN NGHỊ GIỮ',
                'message': f'Model khuyên nên đợi thêm với độ tin cậy {confidence:.1f}%',
                'details': [
                    f'RSI: {latest["rsi"]:.1f} - {"Oversold" if latest["rsi"] < 30 else "Neutral" if latest["rsi"] < 70 else "Overbought"}',
                    f'MACD: {latest["macd"]:.2f} (Signal: {latest["macd_signal"]:.2f})',
                    f'Bollinger: Giá ở {"dưới" if latest["close"] < latest["bb_middle"] else "trên"} đường giữa',
                    f'Khối lượng: {latest["volume"]:,.0f}'
                ],
                'advice': 'Thị trường chưa rõ xu hướng. Nên chờ tín hiệu rõ ràng hơn trước khi hành động.'
            }
        }
        
        rec = recommendations[action_name]
        rec['risk_level'] = risk_level
        rec['risk_color'] = risk_color
        rec['confidence'] = confidence
        
        return rec
    except Exception as e:
        print(f"❌ Error generating recommendation: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['GET'])
def predict():
    """API endpoint for prediction"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        
        # Fetch real-time data
        df = get_realtime_data(symbol=symbol, interval=interval)
        
        if df is None or len(df) == 0:
            return jsonify({
                'success': False,
                'error': 'Không thể lấy dữ liệu từ Binance'
            })
        
        # Get current state
        state = get_state(df, -1)
        
        if state is None:
            return jsonify({
                'success': False,
                'error': 'Không thể trích xuất state'
            })
        
        # Predict action
        prediction = predict_action(state)
        
        if prediction is None:
            return jsonify({
                'success': False,
                'error': 'Không thể dự đoán'
            })
        
        # Get recommendation
        recommendation = get_recommendation(prediction, df)
        
        # Get latest price data
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        price_change = latest['close'] - previous['close']
        price_change_pct = (price_change / previous['close']) * 100
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'price': {
                'current': float(latest['close']),
                'open': float(latest['open']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'volume': float(latest['volume']),
                'change': float(price_change),
                'change_pct': float(price_change_pct)
            },
            'indicators': {
                'rsi': float(latest['rsi']),
                'macd': float(latest['macd']),
                'signal': float(latest['macd_signal']),
                'bb_upper': float(latest['bb_upper']),
                'bb_middle': float(latest['bb_middle']),
                'bb_lower': float(latest['bb_lower'])
            },
            'prediction': prediction,
            'recommendation': recommendation
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/history', methods=['GET'])
def history():
    """API endpoint for price history"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        limit = int(request.args.get('limit', 100))
        
        # Fetch data
        df = get_realtime_data(symbol=symbol, interval=interval, limit=limit)
        
        if df is None or len(df) == 0:
            return jsonify({
                'success': False,
                'error': 'Không thể lấy dữ liệu'
            })
        
        # Convert to JSON
        history_data = []
        for idx, row in df.iterrows():
            history_data.append({
                'timestamp': idx.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'rsi': float(row['rsi']),
                'macd': float(row['macd']),
                'signal': float(row['macd_signal']),
                'bb_upper': float(row['bb_upper']),
                'bb_lower': float(row['bb_lower'])
            })
        
        return jsonify({
            'success': True,
            'data': history_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/status', methods=['GET'])
def status():
    """API endpoint for system status"""
    return jsonify({
        'success': True,
        'model_loaded': dqn_agent is not None,
        'fetcher_ready': fetcher is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("="*70)
    print("🚀 BITCOIN TRADING PREDICTOR - WEB APPLICATION")
    print("="*70)
    
    # Initialize
    print("\n📦 Loading model...")
    if not load_model():
        print("⚠️ Warning: Model not loaded. Please train model first.")
    
    print("\n🔗 Initializing Binance connection...")
    if not initialize_fetcher():
        print("❌ Error: Cannot initialize Binance fetcher")
        sys.exit(1)
    
    print("\n✅ System ready!")
    print("\n🌐 Starting web server...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
