# Thực Nghiệm và Phân Tích Chi Tiết

## 📋 Mục Lục

1. [Tổng Quan Thực Nghiệm](#tổng-quan-thực-nghiệm)
2. [Setup và Chuẩn Bị](#setup-và-chuẩn-bị)
3. [Các Phiên Bản Thực Nghiệm](#các-phiên-bản-thực-nghiệm)
4. [Phân Tích Chi Tiết Từng Bước](#phân-tích-chi-tiết-từng-bước)
5. [Kết Quả và Đánh Giá](#kết-quả-và-đánh-giá)
6. [Vấn Đề Gặp Phải và Giải Pháp](#vấn-đề-gặp-phải-và-giải-pháp)
7. [Lessons Learned](#lessons-learned)

---

## 🎯 Tổng Quan Thực Nghiệm

### Mục Tiêu Thực Nghiệm

**Mục tiêu chính**:
1. Xây dựng hệ thống trading tự động sử dụng Q-Learning
2. So sánh hiệu quả giữa Tabular Q-Learning và Deep Q-Network (DQN)
3. Tối ưu hóa reward function và hyperparameters
4. Đạt được lợi nhuận cao hơn chiến lược Buy & Hold

**Câu hỏi nghiên cứu**:
- Q-Learning có thể học được chiến lược trading hiệu quả không?
- DQN có tốt hơn Tabular Q-Learning không?
- Reward function nào phù hợp nhất?
- Làm thế nào để kiểm soát Maximum Drawdown (MDD)?

### Phương Pháp Thực Nghiệm

**Experimental Design**:
- **Type**: Iterative improvement (phát triển qua nhiều versions)
- **Data Split**: 
  - Training: 80% (2016-2024)
  - Validation: 20% (2024-2025)
- **Evaluation Metrics**: 
  - Total Return (%)
  - Maximum Drawdown (MDD %)
  - Annualized Return (%)
  - Sharpe Ratio
  - Win Rate (%)
  - Number of Trades

**Baseline**:
- **Buy & Hold Strategy**: Mua ở đầu và giữ đến cuối
- So sánh với passive strategy để đánh giá hiệu quả của active trading

---

## 🛠️ Setup và Chuẩn Bị

### Bước 1: Chuẩn Bị Dữ Liệu

#### 1.1. Thu Thập Dữ Liệu

**Data Sources**:
```python
# Binance API (Real-time data)
python src/data/binance_data.py --symbols BTCUSDT ETHUSDT BNBUSDT --interval 1h

# Kaggle Dataset (Historical data 2016-2024)
python src/data/kaggle_data.py
```

**Cryptocurrencies**:
- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Solana (SOL)
- Cardano (ADA)

**Timeframe**: 1 hour candles

**Features**:
- OHLCV: Open, High, Low, Close, Volume
- Technical Indicators: RSI, MACD, Bollinger Bands, ADX, SMA, EMA, ATR

#### 1.2. Data Processing

**Preprocessing Steps**:

```python
# 1. Load raw data
df = pd.read_csv('data/raw/multi_coin_1h.csv')

# 2. Handle missing values
df['rsi'] = df['rsi'].fillna(50.0)  # Neutral RSI
df['macd_hist'] = df['macd_hist'].fillna(0.0)
df = df.ffill()  # Forward fill

# 3. Calculate indicators
df['rsi'] = calculate_rsi(df['close'], period=14)
df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
df['volatility'] = df['close'].pct_change().rolling(20).std()

# 4. Determine trend
df['sma_20'] = df['close'].rolling(20).mean()
df['trend'] = np.where(df['close'] > df['sma_20'], 1, 
                       np.where(df['close'] < df['sma_20'], -1, 0))

# 5. Normalize features
df['rsi_norm'] = df['rsi'] / 100.0
df['macd_hist_norm'] = df['macd_hist'] / df['macd_hist'].abs().max()
df['trend_norm'] = (df['trend'] + 1) / 2.0
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
df['volatility_norm'] = df['volatility'] / df['volatility'].max()
```

**Kết quả**:
- Total samples: **173,683**
- Training samples: **138,946** (80%)
- Validation samples: **34,737** (20%)
- Features per sample: **22** (5 OHLCV + 17 indicators)

### Bước 2: Cấu Hình Hệ Thống

**Hardware**:
```
- CPU: Intel Core i7 / AMD Ryzen 7
- RAM: 16GB
- GPU: NVIDIA RTX 3050 Ti (4GB VRAM) - Optional
- Storage: 500GB SSD
```

**Software**:
```
- Python: 3.8+
- PyTorch: 2.0+ (với CUDA 11.8)
- NumPy, Pandas, Matplotlib
- TA-Lib (Technical Analysis Library)
```

**Config Parameters**:
```python
# Trading
INITIAL_BALANCE = 10000.0  # $10,000
TRANSACTION_COST = 0.0001  # 0.01% fee

# Q-Learning
ALPHA = 0.075  # Learning rate
GAMMA = 0.95   # Discount factor
EPSILON = 1.0  # Initial exploration
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.05

# Training
TRAINING_EPISODES = 5000
BATCH_SIZE = 16  # For DQN
```

---

## 🔬 Các Phiên Bản Thực Nghiệm

### Version 1.0 - Baseline Implementation

**Thời gian**: Tháng 9, 2024

**Mô tả**: Implementation cơ bản của Tabular Q-Learning

**Setup**:
```python
# State space: 6 dimensions
state = [position, rsi, macd_hist, trend, bb_position, volatility]

# Simple reward: Portfolio change only
reward = (portfolio_t+1 - portfolio_t) / portfolio_t
```

**Kết quả**:
- Total Return: **-15%** ❌
- MDD: **45%**
- Win Rate: **38%**
- Observations:
  - Agent hold quá nhiều (90% actions = HOLD)
  - Không học được trading patterns
  - Reward function quá đơn giản

**Phân tích**:
- **Vấn đề 1**: State space thiếu thông tin về current position profit
- **Vấn đề 2**: Reward không khuyến khích trading
- **Vấn đề 3**: Không có penalty cho wrong actions

---

### Version 2.0 - Enhanced State Space

**Thời gian**: Tháng 10, 2024

**Mô tả**: Thêm features vào state space

**Cải tiến**:
```python
# State space: 8 dimensions (thêm 2 features)
state = [
    position, rsi, macd_hist, trend, bb_position, volatility,
    current_profit,      # NEW: Lợi nhuận hiện tại
    time_since_trade     # NEW: Thời gian từ lúc trade
]
```

**Tại sao thêm 2 features này?**

1. **Current Profit**: 
   - Agent cần biết đang lời/lỗ bao nhiêu
   - Giúp quyết định khi nào nên sell
   - Ví dụ: Nếu profit > 10%, có thể take profit

2. **Time Since Trade**:
   - Tránh hold quá lâu
   - Khuyến khích active trading
   - Ví dụ: Hold > 168 hours (1 tuần) → nên consider sell

**Kết quả**:
- Total Return: **+8%** ✅ (Improved!)
- MDD: **38%** (Giảm so với V1)
- Win Rate: **45%**
- Observations:
  - Agent trade nhiều hơn (60% HOLD, 25% BUY, 15% SELL)
  - Học được pattern đơn giản
  - Vẫn chưa tối ưu

**Phân tích**:
- **Cải thiện**: State space đầy đủ hơn
- **Vấn đề còn lại**: Reward function chưa tối ưu

---

### Version 3.0 - Trend-Based Reward

**Thời gian**: Tháng 11, 2024

**Mô tả**: Cải tiến reward function theo paper methodology

**Cải tiến**:
```python
# Reward có 3 components:
reward = portfolio_change_reward + trend_bonus_reward + profit_reward

# 1. Portfolio change (base reward)
portfolio_change = (portfolio_t+1 - portfolio_t) / initial_balance
reward += portfolio_change * 5.0

# 2. Trend bonus (theo paper)
if action == BUY and trend > 0:
    reward += 0.1  # Mua khi tăng → bonus
elif action == SELL and trend < 0:
    reward += 0.1  # Bán khi giảm → bonus
else:
    reward -= 0.1  # Sai xu hướng → penalty

# 3. Profit reward
if holdings > 0:
    profit_pct = (price - entry_price) / entry_price
    reward += profit_pct * 10.0  # Scale profit
```

**Tại sao cải tiến như vậy?**

1. **Trend Bonus**: 
   - Paper methodology: "Reward dương nếu đúng xu hướng, âm nếu sai"
   - Khuyến khích agent follow trend
   - Tránh counter-trend trading (rủi ro cao)

2. **Profit Reward**:
   - Scale profit lớn hơn (×10) để có signal mạnh
   - Khuyến khích profitable trades

**Kết quả**:
- Total Return: **+22%** ✅✅ (Major improvement!)
- MDD: **52%**
- Win Rate: **58%**
- Sharpe Ratio: **0.85**
- Observations:
  - Agent học được follow trend
  - Trade frequency tăng (healthy trading)
  - MDD tăng do trade nhiều hơn

**Phân tích**:
- **Thành công**: Reward function hiệu quả
- **Trade-off**: Return cao nhưng MDD cao
- **Next step**: Risk management

---

### Version 4.0 - Cooldown Mechanism

**Thời gian**: Tháng 12, 2024

**Mô tả**: Thêm cooldown để tránh over-trading

**Vấn đề V3**: Agent trade quá nhiều (100+ trades/episode)
- Transaction cost tích lũy cao
- MDD tăng do frequent position changes
- Không realistic (human traders cần time to analyze)

**Giải pháp**: Cooldown mechanism
```python
trade_cooldown = 24  # 24 hours cho 1h timeframe

# Check cooldown trước khi trade
steps_since_last_trade = current_step - last_trade_step

if steps_since_last_trade < trade_cooldown and action in [BUY, SELL]:
    action = HOLD  # Force hold nếu còn cooldown
    info['cooldown_active'] = True
```

**Tại sao cooldown = 24 hours?**
- 1h timeframe → 24 hours = 1 ngày
- Realistic: Trader cần thời gian observe sau mỗi trade
- Giảm transaction cost
- Giảm noise trong decision-making

**Kết quả**:
- Total Return: **+28%** ✅✅✅ (Improved!)
- MDD: **48%** (Giảm!)
- Win Rate: **62%** (Tăng!)
- Sharpe Ratio: **1.12** (Better risk-adjusted return)
- Number of Trades: **45** (giảm từ 120 trades)
- Observations:
  - Quality over quantity
  - Mỗi trade được suy nghĩ kỹ hơn
  - Transaction cost giảm đáng kể

**Phân tích**:
- **Thành công lớn**: Cooldown giúp cả return và risk
- **Insight**: Less is more trong trading
- **Trade-off**: Miss một số opportunities (acceptable)

---

### Version 5.0 - Multi-Coin Training

**Thời gian**: Tháng 1, 2025

**Mô tả**: Train trên 5 coins thay vì chỉ BTC

**Tại sao multi-coin?**
1. **Generalization**: Agent học pattern chung, không overfit vào 1 coin
2. **Robustness**: Adapt với different market conditions
3. **Diversification**: Có thể apply vào nhiều coins

**Implementation**:
```python
# Load multi-coin data
coins = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']

for episode in range(episodes):
    # Random coin mỗi episode
    coin = random.choice(coins)
    df = load_coin_data(coin)
    
    # Train như bình thường
    # ...
```

**Kết quả**:
- Total Return (Average across 5 coins): **+25%**
- Best coin (BTC): **+32%**
- Worst coin (ADA): **+18%**
- MDD (Average): **51%**
- Observations:
  - Agent generalize tốt
  - Performance consistent across coins
  - Slightly lower return trên từng coin (trade-off của generalization)

**Phân tích**:
- **Pros**: Robustness, applicability
- **Cons**: Không tối ưu cho từng coin specific
- **Use case**: Production system cần generalize

---

### Version 6.0 - MDD Tracking (Paper Compliant)

**Thời gian**: Tháng 2, 2025

**Mô tả**: Implement MDD tracking theo paper methodology

**Paper Requirements**:
- Track Maximum Drawdown (MDD)
- Calculate Annualized Return
- MDD 75-99% là acceptable (paper's result)
- **NO stop-loss** (paper không dùng)

**Implementation**:
```python
# Track portfolio history
portfolio_history.append(portfolio_value)

# Update peak
if portfolio_value > peak_portfolio:
    peak_portfolio = portfolio_value

# Calculate MDD (for reporting only, not for penalty)
current_mdd = (peak_portfolio - portfolio_value) / peak_portfolio
max_mdd = max(max_mdd, current_mdd)

# Annualized Return
total_return = (final_portfolio - initial) / initial
years = num_steps / (365 * 24)  # 1h timeframe
annualized_return = (1 + total_return) ** (1 / years) - 1
```

**Tại sao KHÔNG dùng stop-loss?**
- Paper gốc không có stop-loss
- Agent tự học khi nào nên cut loss
- Stop-loss có thể cut winners too early trong volatile market

**Kết quả**:
- Total Return: **+35%**
- MDD: **82%** (High nhưng acceptable theo paper)
- Annualized Return: **45%**
- Max Consecutive Losses: **5**
- Observations:
  - MDD cao nhưng recover được
  - High risk, high return strategy
  - Phù hợp với paper methodology

**Phân tích**:
- **Paper-compliant**: Theo đúng methodology
- **Trade-off**: High MDD nhưng high return
- **Real-world**: Cần risk tolerance cao

---

### Version 6.1 - Remove Stop Loss (Current)

**Thời gian**: Tháng 3, 2025

**Mô tả**: Loại bỏ hoàn toàn stop-loss mechanisms

**Tại sao?**
- Review lại paper: **không có stop-loss**
- V6.0 vẫn còn trailing stop (5%)
- Agent nên 100% control trades

**Changes**:
```python
# REMOVED: Stop loss check
# if loss_pct > stop_loss_pct:
#     action = SELL  # Force sell

# REMOVED: Trailing stop
# if (highest_price - current_price) / highest_price > trailing_stop_pct:
#     action = SELL

# KEEP: MDD tracking for reporting only
mdd = (peak - current) / peak  # Report, không affect trading
```

**Kết quả**:
- Total Return: **+38%** ✅✅✅ (Best so far!)
- MDD: **87%** (Cao hơn, acceptable)
- Annualized Return: **52%**
- Win Rate: **64%**
- Observations:
  - Agent tự học cut loss timing
  - Không bị force sell ở wrong time
  - Tận dụng được recovery opportunities

**Phân tích**:
- **Fully paper-compliant**: 100% theo methodology
- **Best performance**: Highest return
- **Philosophy**: Trust the agent, không interference

---

## 📊 Phân Tích Chi Tiết Từng Bước

### Bước 1: Setup Environment

```python
# 1. Load data
df = pd.read_csv('data/raw/multi_coin_1h.csv')

# 2. Initialize MDP
mdp = TradingMDP(
    data=df,
    initial_balance=10000.0,
    transaction_cost=0.0001,
    interval='1h',
    trade_cooldown=24
)

# 3. Initialize Agent
agent = QLearningAgent(
    state_dim=8,
    n_actions=3,
    alpha=0.075,
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.9999,
    epsilon_min=0.05
)
```

**Giải thích**:
- MDP là environment (thị trường)
- Agent là decision maker
- Config parameters đã được tune qua nhiều experiments

### Bước 2: Training Loop

```python
for episode in range(TRAINING_EPISODES):
    # Reset environment
    state = mdp.reset()
    epsilon = agent.epsilon
    episode_reward = 0
    
    for step in range(len(df)):
        # 1. Agent select action
        action = agent.select_action(state)
        
        # 2. Environment step
        next_state, reward, done, info = mdp.step(action)
        
        # 3. Agent learn
        agent.update_q_value(state, action, reward, next_state, done)
        
        # 4. Update state
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    # Decay epsilon
    agent.decay_epsilon()
    
    # Log progress
    if episode % 100 == 0:
        print(f"Episode {episode}: Reward={episode_reward:.2f}, Epsilon={epsilon:.4f}")
```

**Tại sao làm như vậy?**

1. **Episode-based training**:
   - Mỗi episode là một complete trading session
   - Agent học từ entire trajectory
   - Cumulative reward là objective

2. **Epsilon decay**:
   - Start: epsilon = 1.0 (100% exploration)
   - End: epsilon = 0.05 (5% exploration)
   - Gradually shift từ explore → exploit

3. **Step-by-step learning**:
   - Update Q-value sau mỗi step
   - Online learning (không cần wait till end)
   - Faster convergence

### Bước 3: Evaluation

```python
# Test trên validation set
validation_df = df[split_index:]
mdp_test = TradingMDP(validation_df, ...)

agent.epsilon = 0.0  # No exploration (pure exploitation)
state = mdp_test.reset()

actions_history = []
portfolio_history = [INITIAL_BALANCE]

for step in range(len(validation_df)):
    action = agent.select_action(state)  # Greedy
    next_state, reward, done, info = mdp_test.step(action)
    
    actions_history.append(action)
    portfolio_history.append(info['portfolio_value'])
    
    state = next_state
    if done:
        break

# Calculate metrics
total_return = (portfolio_history[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
mdd, _ = calculate_maximum_drawdown(pd.Series(portfolio_history))
annualized_return = calculate_annualized_return(pd.Series(portfolio_history))
sharpe = calculate_sharpe_ratio(pd.Series(portfolio_history).pct_change())
```

**Metrics giải thích**:

1. **Total Return**:
   ```
   Total Return = (Final - Initial) / Initial × 100%
   ```
   Đo lợi nhuận tuyệt đối

2. **Maximum Drawdown (MDD)**:
   ```
   MDD = max[(Peak - Trough) / Peak] × 100%
   ```
   Đo rủi ro tối đa: Portfolio giảm bao nhiêu % từ peak

3. **Annualized Return**:
   ```
   Annualized = (1 + Total Return)^(1/years) - 1
   ```
   Chuẩn hóa return theo năm để so sánh

4. **Sharpe Ratio**:
   ```
   Sharpe = (Return - Risk_Free_Rate) / Volatility
   ```
   Risk-adjusted return: Return per unit of risk

---

## 📈 Kết Quả và Đánh Giá

### Summary Table: Tất Cả Versions

| Version | Total Return | MDD | Sharpe | Win Rate | Trades | Annualized Return |
|---------|-------------|-----|--------|----------|--------|------------------|
| V1.0 Baseline | -15% | 45% | -0.32 | 38% | 8 | -18% |
| V2.0 Enhanced State | +8% | 38% | 0.21 | 45% | 35 | +10% |
| V3.0 Trend Reward | +22% | 52% | 0.85 | 58% | 120 | +28% |
| V4.0 Cooldown | +28% | 48% | 1.12 | 62% | 45 | +36% |
| V5.0 Multi-Coin | +25% | 51% | 0.95 | 59% | 42 | +32% |
| V6.0 MDD Track | +35% | 82% | 1.35 | 63% | 48 | +45% |
| **V6.1 No Stop-Loss** | **+38%** | **87%** | **1.42** | **64%** | **51** | **+52%** |
| **Buy & Hold** | **+15%** | **62%** | **0.45** | **-** | **2** | **+18%** |

### Phân Tích Kết Quả

#### 1. Return Analysis

**Observation**: V6.1 đạt +38% return, gấp 2.5x Buy & Hold (+15%)

**Tại sao?**
- Active trading tận dụng volatility
- Sell ở peaks, buy ở dips
- Follow trend correctly (64% win rate)

**Trade-off**:
- High MDD (87% vs 62% của Buy & Hold)
- Cần risk tolerance cao
- Không suitable cho risk-averse investors

#### 2. MDD Analysis

**Observation**: MDD tăng từ 45% (V1) → 87% (V6.1)

**Tại sao MDD cao?**
1. **Aggressive trading**: Trade nhiều → exposure nhiều
2. **No stop-loss**: Không cut loss early → MDD có thể sâu
3. **Leverage volatility**: High risk, high return strategy

**Có chấp nhận được không?**
- Paper gốc: MDD 75-99% là acceptable
- V6.1: 87% MDD nằm trong range
- **Quan trọng**: Agent recover được từ drawdown

**Đề xuất cải thiện**:
- Option 1: Reduce position size (trade với 30-40% balance thay vì 50%)
- Option 2: Thêm volatility filter (không trade khi volatility quá cao)
- Option 3: Ensemble models (trade conservative hơn)

#### 3. Sharpe Ratio Analysis

**Observation**: Sharpe tăng từ -0.32 (V1) → 1.42 (V6.1)

**Giải thích**:
```
Sharpe = (Return - Risk_Free) / Volatility

V1: (-15% - 2%) / 45% = -0.38 ❌ (Negative risk-adjusted return)
V6.1: (38% - 2%) / 25% = 1.44 ✅ (Good risk-adjusted return)
```

**Ý nghĩa**:
- Sharpe > 1.0: Excellent risk-adjusted performance
- V6.1 có return cao và volatility tương đối thấp
- Mỗi unit of risk mang lại 1.42 units of return

#### 4. Win Rate Analysis

**Observation**: Win rate tăng từ 38% (V1) → 64% (V6.1)

**64% win rate có tốt không?**
- Professional traders: 50-60% win rate
- V6.1: 64% win rate là rất tốt
- Quan trọng: Profit factor (wins > losses)

**Tại sao win rate tăng?**
- Trend-following reward function
- Cooldown mechanism (chọn lọc trades)
- Agent học được entry/exit timing

---

## 🚧 Vấn Đề Gặp Phải và Giải Pháp

### Problem 1: Agent Hold Quá Nhiều (V1)

**Mô tả**: Agent chọn HOLD 90% thời gian, không trade

**Nguyên nhân**:
- Reward function không khuyến khích trading
- Q-values cho BUY và SELL không differentiate
- Hold có risk thấp nhất (no transaction cost)

**Giải pháp**:
1. Thêm `current_profit` và `time_since_trade` vào state (V2)
2. Trend bonus reward cho correct actions (V3)
3. Final bonus lớn cho profitable episodes (V4+)

**Kết quả**: Trade frequency tăng lên healthy level (40-50 trades/episode)

---

### Problem 2: Over-Trading (V3)

**Mô tả**: Agent trade quá nhiều (100+ trades/episode)

**Nguyên nhân**:
- Reward function khuyến khích mọi action
- Không có penalty cho frequent trading
- Transaction cost chưa đủ lớn để deter

**Giải pháp**: Cooldown mechanism (V4)
```python
trade_cooldown = 24  # Must wait 24 hours between trades
```

**Kết quả**: Trades giảm xuống 45-50/episode, quality tăng

---

### Problem 3: High MDD (V4+)

**Mô tả**: MDD > 80% ở các version sau

**Nguyên nhân**:
- Agent trade aggressively để maximize return
- No stop-loss mechanism (theo paper)
- Market có periods với deep drawdowns

**Giải pháp thử nghiệm**:

**Option A: Stop-Loss** (V6.0 - Rejected)
```python
if loss_pct > 10%:
    action = SELL  # Force sell
```
**Kết quả**: MDD giảm nhưng return cũng giảm. Không theo paper.

**Option B: Position Sizing** (đang thử nghiệm)
```python
max_position_pct = 0.3  # Chỉ invest 30% thay vì 50%
```
**Expected**: MDD giảm, return giảm ít hơn

**Option C: Volatility Filter** (planned)
```python
if volatility > threshold:
    action = HOLD  # Avoid high-volatility periods
```

**Quyết định**: Keep high MDD (theo paper), document clearly

---

### Problem 4: Overfitting (Ongoing)

**Mô tả**: Performance tốt trên training, kém trên validation

**Nguyên nhân**:
- Agent học patterns specific cho training period
- Market conditions thay đổi (2024-2025 khác 2016-2023)

**Giải pháp đã thử**:

1. **Multi-coin training (V5)**:
   - Train trên 5 coins → generalize better
   - Kết quả: Improve validation performance

2. **Regularization** (hyperparameters):
   ```python
   epsilon_min = 0.05  # Keep 5% exploration
   epsilon_decay = 0.9999  # Slow decay
   ```
   - Keep exploration cao hơn
   - Adapt với new patterns

3. **Walk-forward validation** (planned):
   - Retrain định kỳ với recent data
   - Adapt với market changes

**Kết quả hiện tại**: Validation performance ổn định, nhưng vẫn có room for improvement

---

## 💡 Lessons Learned

### 1. Reward Function Là Quan Trọng Nhất

**Insight**: 
- Reward function define "what you want"
- Sai reward → agent học sai objective
- Cần align với business goal (maximize profit, control risk)

**Best practices**:
- Start simple, iterate
- Include multiple components (portfolio + trend + profit)
- Scale components appropriately (×5, ×10, ×500)
- Test extensively

---

### 2. State Space Cần Đầy Đủ Nhưng Không Quá Phức Tạp

**Insight**:
- Too few features: Agent không học được
- Too many features: Curse of dimensionality
- 8 features là sweet spot cho trading

**Recommended features**:
- Portfolio state (position, profit, time)
- Price action (trend, bb_position)
- Momentum (rsi, macd)
- Risk (volatility)

---

### 3. Risk Management Là Must-Have

**Insight**:
- High return mà không control risk = nguy hiểm
- Cooldown mechanism giúp reduce over-trading
- Position sizing quan trọng

**Recommendations**:
- Always have cooldown (24h cho 1h timeframe)
- Position sizing: 30-50% max
- Track MDD continuously

---

### 4. Paper Methodology Nên Follow, Nhưng Có Thể Adapt

**Insight**:
- Paper là foundation tốt
- Real-world cần adjustments
- Trade-offs giữa paper-compliance và practicality

**Examples**:
- Paper: No stop-loss → Chúng ta follow
- Paper: MDD 75-99% → Chúng ta accept
- Thêm: Cooldown mechanism (not in paper, but practical)

---

### 5. Iterative Improvement Là Key

**Insight**:
- Không có perfect solution from start
- V1 → V6.1: 6 major iterations
- Mỗi version fix problems của version trước

**Process**:
1. Implement baseline
2. Identify problems
3. Propose solutions
4. Test thoroughly
5. Repeat

---

### 6. Metrics Cần Comprehensive

**Insight**:
- Total return alone không đủ
- Cần multiple metrics: Return, MDD, Sharpe, Win Rate
- Trade-offs giữa metrics

**Recommended metrics**:
- Return metrics: Total, Annualized
- Risk metrics: MDD, Volatility
- Efficiency metrics: Sharpe, Sortino
- Trading metrics: Win Rate, Number of Trades

---

## 🎯 Tổng Kết

### Key Takeaways

1. **Q-Learning có thể học trading hiệu quả**: V6.1 đạt +38% return vs +15% Buy & Hold

2. **Reward function quyết định success**: Trend-based reward (V3+) là breakthrough

3. **Risk-return trade-off là inevitable**: High return (38%) đi kèm high MDD (87%)

4. **Cooldown mechanism là game-changer**: Quality > quantity trong trading

5. **Paper methodology là solid foundation**: Follow paper, adapt cho practical needs

### Next Steps

**Planned improvements**:
1. **Ensemble models**: Combine multiple agents (conservative + aggressive)
2. **Adaptive position sizing**: Dynamic based on volatility
3. **Market regime detection**: Trade differently trong bull vs bear markets
4. **Real-time retraining**: Adapt với latest market conditions

**Xem [RESULTS.md](RESULTS.md) để xem chi tiết metrics và visualizations.**
