# ĐỀ XUẤT CẢI THIỆN MÔ HÌNH DQN

## Vấn đề hiện tại (từ test 2020-03-12)

Model DQN đã **BUY 70%** trong ngày crash -39.34%, dẫn đến:
- Accuracy: 37.50% (tệ hơn random)
- Loss: -19.67%
- R² Score: 0.0690 (không học được gì)

## Giải pháp đề xuất

### 1. CẢI THIỆN REWARD FUNCTION ⭐⭐⭐

#### Thêm Trend Penalty/Bonus mạnh hơn:

```python
# Trong file: src/models/mdp_trading.py
def calculate_reward(self, action, prev_value, current_value):
    # ... existing code ...
    
    # THÊM: Penalty mạnh khi Buy ngược trend
    if action == 1:  # Buy
        if self.df.iloc[self.current_step]['trend'] < -0.5:
            reward -= 0.5  # Penalty lớn nếu Buy trong downtrend mạnh
        if self.df.iloc[self.current_step]['rsi'] < 30:
            # Tuy oversold nhưng trend vẫn giảm
            if self.df.iloc[self.current_step]['trend'] < 0:
                reward -= 0.3  # Avoid "catch falling knife"
    
    # THÊM: Bonus khi Sell/Hold trong bear market
    if action in [0, 2]:  # Hold or Sell
        if self.df.iloc[self.current_step]['trend'] < -0.5:
            reward += 0.2  # Thưởng khi tránh được downtrend
    
    # THÊM: Momentum penalty
    # Check 3 candles trước có đang giảm liên tục không
    if self.current_step >= 3:
        recent_prices = [self.df.iloc[i]['close'] 
                        for i in range(self.current_step-3, self.current_step)]
        is_downtrend = all(recent_prices[i] > recent_prices[i+1] 
                          for i in range(len(recent_prices)-1))
        
        if is_downtrend and action == 1:  # Buy during continuous drop
            reward -= 0.4
    
    return reward
```

### 2. RE-TRAIN VỚI BALANCED DATA ⭐⭐

```python
# Tạo training data có đủ bull/bear/sideways
# File: src/models/enhanced_training_gpu.py

def create_balanced_training_data(df, train_ratio=0.8):
    """
    Chia data thành 3 nhóm: Bull, Bear, Sideways
    Đảm bảo mỗi nhóm có đủ samples
    """
    df['daily_return'] = df.groupby('coin')['close'].pct_change(24)
    
    # Classify market regime
    df['regime'] = 'sideways'
    df.loc[df['daily_return'] > 0.02, 'regime'] = 'bull'
    df.loc[df['daily_return'] < -0.02, 'regime'] = 'bear'
    
    # Balance samples
    min_samples = df.groupby('regime').size().min()
    
    balanced_df = pd.concat([
        df[df['regime'] == 'bull'].sample(min_samples),
        df[df['regime'] == 'bear'].sample(min_samples),
        df[df['regime'] == 'sideways'].sample(min_samples)
    ])
    
    return balanced_df.sample(frac=1).reset_index(drop=True)
```

### 3. THÊM STOP-LOSS ĐỘNG ⭐⭐⭐

```python
# Trong TradingMDP
class TradingMDP:
    def __init__(self, ...):
        # ... existing code ...
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.entry_price = None
    
    def step(self, action):
        # ... existing code ...
        
        # Check stop loss
        if self.position == 1 and self.entry_price is not None:
            current_price = self.df.iloc[self.current_step]['close']
            loss_pct = (current_price - self.entry_price) / self.entry_price
            
            if loss_pct < -self.stop_loss_pct:
                # Force sell
                action = 2  # Override to Sell
                reward -= 1.0  # Penalty for hitting stop loss
        
        # Track entry price
        if action == 1:  # Buy
            self.entry_price = self.df.iloc[self.current_step]['close']
        elif action == 2:  # Sell
            self.entry_price = None
```

### 4. THÊM MARKET REGIME DETECTOR ⭐⭐

```python
# Thêm feature: Detect market regime
def detect_market_regime(df, lookback=24):
    """
    Detect: Bull, Bear, or Sideways
    Based on: Price trend, Volatility, Volume
    """
    df['regime_score'] = 0.0
    
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        
        # Price trend
        price_change = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
        
        # Trend consistency
        trend_consistency = window['trend'].mean()
        
        # Volatility
        volatility = window['volatility'].mean()
        
        # Score
        if price_change > 0.05 and trend_consistency > 0.3:
            regime = 'bull'
            score = 1.0
        elif price_change < -0.05 and trend_consistency < -0.3:
            regime = 'bear'
            score = -1.0
        else:
            regime = 'sideways'
            score = 0.0
        
        df.loc[i, 'regime_score'] = score
    
    return df

# Thêm vào state
state = [
    position,
    rsi,
    macd_hist,
    trend,
    bb_position,
    volatility,
    price_change,
    profit_pct,
    regime_score  # NEW: Market regime indicator
]
```

### 5. TEST VỚI NHIỀU NGÀY KHÁC NHAU ⭐

Không chỉ test 1 ngày. Cần test:

```python
# Test suite
test_dates = [
    '2020-03-12',  # Strong Bear (-39%)
    '2020-04-22',  # Strong Bull (+4.3%)
    '2024-09-30',  # Sideways
    '2021-05-19',  # Bear (-13%)
    '2020-04-06',  # Bull (+6.7%)
]

for date in test_dates:
    predictor = OneDayPredictor(...)
    results = predictor.run(test_date=date)
    # Compare results
```

### 6. SỬ DỤNG CLASSIFICATION METRICS ⭐

Thay vì MAE/MSE/RMSE (cho regression), dùng:

```python
from sklearn.metrics import classification_report, confusion_matrix

# Ideal actions vs Predicted actions
print(classification_report(ideal_actions, pred_actions, 
                          target_names=['Hold', 'Buy', 'Sell']))

# Confusion matrix
cm = confusion_matrix(ideal_actions, pred_actions)
print("Confusion Matrix:")
print(cm)
```

## Thứ tự ưu tiên

1. **CẢI THIỆN REWARD FUNCTION** (quan trọng nhất)
2. **THÊM STOP-LOSS ĐỘNG** (bảo vệ vốn)
3. **RE-TRAIN VỚI BALANCED DATA** (học đủ các regime)
4. **THÊM MARKET REGIME DETECTOR** (hiểu context)
5. **TEST NHIỀU NGÀY** (validate tổng thể)

## Kết quả mong đợi

Sau khi cải thiện:
- **Accuracy > 60%** (hiện tại: 37.5%)
- **Loss < 10%** trong crash days (hiện tại: -19.67%)
- **Buy actions < 30%** trong bear market (hiện tại: 70%)
- **Sell/Hold actions > 70%** trong bear market

## Timeline

- **Tuần 1**: Cải thiện reward function + re-train
- **Tuần 2**: Thêm stop-loss + market regime detector
- **Tuần 3**: Test và fine-tune
- **Tuần 4**: Validation trên nhiều ngày

---
**Ngày tạo**: 2025-12-17
**Dựa trên**: Test results từ 2020-03-12 (COVID Crash)
