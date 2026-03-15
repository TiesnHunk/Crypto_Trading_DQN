"""
Quick test script to verify everything works
Script test nhanh để kiểm tra hệ thống hoạt động
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("QUICK TEST - KIỂM TRA HỆ THỐNG")
print("="*70)

# Test 1: Import modules
print("\n[1] Kiểm tra imports...")
try:
    from binance_data import BinanceDataFetcher
    from indicators import TechnicalIndicators
    from mdp_trading import TradingMDP
    from q_learning import QLearningAgent
    from metrics import TradingMetrics
    from trend_trading import TrendBasedStrategy
    print("✅ Tất cả modules đã import thành công")
except Exception as e:
    print(f"❌ Lỗi import: {e}")
    exit(1)

# Test 2: Load config
print("\n[2] Kiểm tra config...")
try:
    from config import BINANCE_API_KEY, BINANCE_API_SECRET
    print("✅ Đã load API credentials từ config.py")
    print(f"   API Key: {BINANCE_API_KEY[:10]}...")
except ImportError:
    print("⚠️ Không tìm thấy config.py (OK nếu dùng public API)")

# Test 3: Binance connection
print("\n[3] Kiểm tra kết nối Binance...")
try:
    fetcher = BinanceDataFetcher()
    
    # Test với dữ liệu nhỏ
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    print("   Đang lấy dữ liệu test...")
    df = fetcher.get_klines('BTCUSDT', '1h', start_date, end_date, limit=10)
    
    if df is not None and len(df) > 0:
        print(f"✅ Kết nối Binance thành công! ({len(df)} candles)")
    else:
        print("⚠️ Không lấy được dữ liệu (có thể dùng dữ liệu mẫu)")
except Exception as e:
    print(f"⚠️ Không kết nối được Binance: {e}")
    print("   Sẽ dùng dữ liệu mẫu thay thế")

# Test 4: Create sample data
print("\n[4] Tạo dữ liệu mẫu...")
dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
np.random.seed(42)
prices = 30000 + np.cumsum(np.random.randn(100) * 100)

df = pd.DataFrame({
    'open': prices + np.random.randn(100) * 50,
    'high': prices + np.abs(np.random.randn(100) * 50),
    'low': prices - np.abs(np.random.randn(100) * 50),
    'close': prices,
    'volume': np.random.randint(1000000, 10000000, 100)
}, index=dates)

print(f"✅ Đã tạo dữ liệu mẫu: {len(df)} candles")

# Test 5: Calculate indicators
print("\n[5] Tính toán indicators...")
try:
    df = TechnicalIndicators.add_all_indicators(df)
    df = df.dropna()
    print(f"✅ Đã tính toán indicators: {len(df)} rows")
except Exception as e:
    print(f"❌ Lỗi tính indicators: {e}")
    exit(1)

# Test 6: Create MDP
print("\n[6] Tạo MDP environment...")
try:
    mdp = TradingMDP(df, initial_balance=10000)
    state = mdp.reset()
    print(f"✅ MDP created: state dimension = {len(state)}")
except Exception as e:
    print(f"❌ Lỗi tạo MDP: {e}")
    exit(1)

# Test 7: Create and test agent
print("\n[7] Test Q-Learning agent...")
try:
    agent = QLearningAgent(state_dim=6, n_actions=3, discrete=True)
    
    # Quick training
    print("   Training 10 episodes...")
    agent.train(mdp, n_episodes=10, verbose=False)
    
    # Test
    final_value, return_rate, _ = agent.test(mdp)
    print(f"✅ Agent test thành công!")
    print(f"   Final value: ${final_value:.2f}")
    print(f"   Return: {return_rate*100:.2f}%")
except Exception as e:
    print(f"❌ Lỗi agent: {e}")
    exit(1)

# Test 8: Metrics
print("\n[8] Test metrics...")
try:
    from metrics import TradingMetrics
    
    # Get trading history
    _, _, history = agent.test(mdp)
    portfolio_values = history['portfolio_value']
    
    metrics = TradingMetrics.calculate_all_metrics(history, portfolio_values)
    print(f"✅ Metrics calculated:")
    print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
except Exception as e:
    print(f"❌ Lỗi metrics: {e}")
    exit(1)

# Summary
print("\n" + "="*70)
print("KẾT QUẢ TEST")
print("="*70)
print("✅ TẤT CẢ TEST ĐÃ PASS!")
print("\nHệ thống sẵn sàng sử dụng:")
print("  - Chạy 'python main.py' để train và backtest đầy đủ")
print("  - Kiểm tra README.md để xem hướng dẫn chi tiết")
print("="*70)

