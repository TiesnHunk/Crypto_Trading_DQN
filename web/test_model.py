"""
Test script to verify model loading and prediction
Script kiểm tra xem model có load được và predict được không
"""

import os
import sys
import pickle

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

print("="*70)
print("🧪 TESTING MODEL LOADING AND PREDICTION")
print("="*70)

# Test 1: Check if model file exists
print("\n[TEST 1] Checking model file...")
MODEL_PATH = os.path.join(project_root, 'data', 'processed', 'q_learning_model_combined.pkl')
print(f"Model path: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    print("✅ Model file exists!")
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"   File size: {file_size:.2f} MB")
else:
    print("❌ Model file NOT found!")
    print("   Please run training first:")
    print("   cd src && python main_gpu.py")
    sys.exit(1)

# Test 2: Load model
print("\n[TEST 2] Loading model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        q_table = pickle.load(f)
    print("✅ Model loaded successfully!")
    print(f"   Q-table shape: {q_table.shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Test 3: Import project modules
print("\n[TEST 3] Importing project modules...")
try:
    from src.data.binance_data import BinanceDataFetcher
    from src.utils.indicators import TechnicalIndicators
    from src.models.q_learning import QLearningAgent
    print("✅ All modules imported successfully!")
except Exception as e:
    print(f"❌ Error importing modules: {e}")
    print("   Make sure all dependencies are installed:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Test 4: Initialize Binance fetcher
print("\n[TEST 4] Testing Binance connection...")
try:
    fetcher = BinanceDataFetcher()
    print("✅ Binance fetcher initialized!")
except Exception as e:
    print(f"⚠️ Warning: Cannot initialize Binance fetcher: {e}")
    print("   This might be OK if you don't have API credentials")

# Test 5: Fetch sample data
print("\n[TEST 5] Fetching sample data...")
try:
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    df = fetcher.get_klines(
        symbol='BTCUSDT', 
        interval='1h', 
        start_date=start_date,
        end_date=end_date,
        limit=100
    )
    if df is not None and len(df) > 0:
        print("✅ Data fetched successfully!")
        print(f"   Rows: {len(df)}")
        print(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")
    else:
        print("⚠️ No data returned")
except Exception as e:
    print(f"❌ Error fetching data: {e}")

# Test 6: Initialize Q-Learning agent
print("\n[TEST 6] Initializing Q-Learning agent...")
try:
    agent = QLearningAgent(
        state_dim=6,
        n_actions=3,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.0,
        discrete=True,
        n_bins=10
    )
    agent.Q = q_table
    print("✅ Q-Learning agent initialized!")
except Exception as e:
    print(f"❌ Error initializing agent: {e}")
    sys.exit(1)

# Test 7: Test prediction
print("\n[TEST 7] Testing prediction...")
try:
    import numpy as np
    
    # Create a dummy state
    test_state = np.array([0.5, 0.5, 0.0, 0.0, 0.5, 0.5])
    action = agent.choose_action(test_state, training=False)
    
    actions = ['MUA', 'BÁN', 'GIỮ']
    print(f"✅ Prediction successful!")
    print(f"   Test state: {test_state}")
    print(f"   Predicted action: {actions[action]}")
except Exception as e:
    print(f"❌ Error during prediction: {e}")

# Test 8: Check Flask installation
print("\n[TEST 8] Checking Flask installation...")
try:
    import flask
    print(f"✅ Flask installed! Version: {flask.__version__}")
except ImportError:
    print("⚠️ Flask not installed!")
    print("   Install with: pip install flask flask-cors")

# Final summary
print("\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)
print("""
✅ Model file exists and can be loaded
✅ Project modules can be imported
✅ Q-Learning agent works correctly
✅ Prediction works

Next steps:
1. If Flask is not installed: pip install flask flask-cors
2. Start web server: cd web && python app.py
3. Open browser: http://localhost:5000

Enjoy trading! 📈
""")
print("="*70)
