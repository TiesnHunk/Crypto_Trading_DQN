# Phương Pháp Nghiên Cứu Chi Tiết

## 📋 Mục Lục

1. [Tổng Quan Phương Pháp](#tổng-quan-phương-pháp)
2. [Markov Decision Process (MDP)](#markov-decision-process-mdp)
3. [Không Gian Trạng Thái (State Space)](#không-gian-trạng-thái-state-space)
4. [Không Gian Hành Động (Action Space)](#không-gian-hành-động-action-space)
5. [Hàm Thưởng và Phạt (Reward & Penalty Function)](#hàm-thưởng-và-phạt-reward--penalty-function)
6. [Thuật Toán Q-Learning](#thuật-toán-q-learning)
7. [Deep Q-Network (DQN)](#deep-q-network-dqn)
8. [Technical Indicators](#technical-indicators)
9. [Risk Management](#risk-management)

---

## 🎯 Tổng Quan Phương Pháp

### Bài Toán Trading Như Một MDP

Trading cryptocurrency được mô hình hóa như một **Markov Decision Process (MDP)** với các thành phần:

- **Agent**: Hệ thống trading (Q-Learning/DQN)
- **Environment**: Thị trường cryptocurrency
- **State**: Tình trạng hiện tại của thị trường và portfolio
- **Action**: Quyết định giao dịch (Buy/Sell/Hold)
- **Reward**: Lợi nhuận/lỗ từ giao dịch
- **Goal**: Tối đa hóa tổng reward tích lũy (cumulative reward)

### Framework Tổng Quan

```
┌─────────────────────────────────────────────────────────────┐
│                    Trading MDP Framework                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐     │
│  │  State   │───────▶│  Agent   │───────▶│  Action  │     │
│  │ (market) │        │(Q-Learn) │        │(Buy/Sell)│     │
│  └──────────┘        └──────────┘        └──────────┘     │
│       ▲                                         │          │
│       │                                         ▼          │
│  ┌────┴─────┐                          ┌─────────────┐    │
│  │  Reward  │◀─────────────────────────│Environment  │    │
│  │(+profit) │                          │  (Market)   │    │
│  └──────────┘                          └─────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Markov Decision Process (MDP)

### Định Nghĩa Chính Thức

MDP được định nghĩa bởi tuple: **M = (S, A, P, R, γ)**

Trong đó:
- **S**: State space (không gian trạng thái)
- **A**: Action space (không gian hành động)
- **P**: Transition probability P(s'|s,a) (xác suất chuyển trạng thái)
- **R**: Reward function R(s,a,s') (hàm thưởng)
- **γ**: Discount factor (hệ số chiết khấu, γ ∈ [0,1])

### Áp Dụng Vào Trading

| MDP Component | Trading Application |
|--------------|---------------------|
| **State (s)** | Thông tin thị trường + Portfolio (giá, indicators, position) |
| **Action (a)** | Buy, Sell, Hold |
| **Transition P(s'\|s,a)** | Giá thị trường sau khi thực hiện action |
| **Reward R(s,a,s')** | Profit/Loss từ action đó |
| **Discount γ** | γ = 0.95 (ưu tiên reward ngắn hạn hơn) |

### Markov Property

**Giả định**: Trạng thái hiện tại **s_t** chứa đủ thông tin để dự đoán tương lai, không cần lịch sử:

```
P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0)
```

**Giải thích**: State hiện tại (bao gồm indicators như RSI, MACD) đã tóm tắt xu hướng quá khứ.

---

## 🧩 Không Gian Trạng Thái (State Space)

### Định Nghĩa State Space

State là vector **8 chiều** mô tả tình trạng thị trường và portfolio:

```python
state = [
    position,           # 0: không giữ, 1: đang giữ coin
    rsi_norm,           # RSI normalized (0-1)
    macd_hist_norm,     # MACD histogram normalized
    trend_norm,         # Xu hướng giá: 0 (giảm), 0.5 (ngang), 1 (tăng)
    bb_position,        # Vị trí giá trong Bollinger Bands (0-1)
    volatility_norm,    # Volatility normalized
    current_profit,     # Lợi nhuận hiện tại nếu đang hold (normalized)
    time_since_trade    # Thời gian từ lúc mua (normalized)
]
```

### Giải Thích Từng Thành Phần

#### 1. Position (Vị thế giao dịch)
- **Giá trị**: 0 hoặc 1
- **Ý nghĩa**: 
  - 0 = Không giữ coin (all cash)
  - 1 = Đang giữ coin (holding position)
- **Mục đích**: Agent biết trạng thái hiện tại để quyết định hành động phù hợp

#### 2. RSI (Relative Strength Index)
- **Công thức**: 
  ```
  RSI = 100 - (100 / (1 + RS))
  RS = Average Gain / Average Loss (14 periods)
  ```
- **Normalization**: `rsi_norm = rsi / 100.0` (0-1)
- **Ý nghĩa**: 
  - RSI > 70: Overbought (quá mua, có thể giảm)
  - RSI < 30: Oversold (quá bán, có thể tăng)

#### 3. MACD Histogram
- **Công thức**: 
  ```
  MACD = EMA(12) - EMA(26)
  Signal = EMA(MACD, 9)
  Histogram = MACD - Signal
  ```
- **Normalization**: Chia cho max absolute value
- **Ý nghĩa**: 
  - Histogram > 0: Xu hướng tăng
  - Histogram < 0: Xu hướng giảm

#### 4. Trend (Xu hướng giá)
- **Tính toán**: So sánh giá với SMA
  ```python
  if price > SMA(20): trend = 1      # Uptrend
  elif price < SMA(20): trend = -1   # Downtrend
  else: trend = 0                     # Sideways
  ```
- **Normalization**: `trend_norm = (trend + 1) / 2.0` → {0, 0.5, 1}

#### 5. Bollinger Bands Position
- **Công thức**: 
  ```
  BB_Upper = SMA(20) + 2*STD(20)
  BB_Lower = SMA(20) - 2*STD(20)
  BB_Position = (Price - BB_Lower) / (BB_Upper - BB_Lower)
  ```
- **Ý nghĩa**: 
  - bb_position ≈ 0: Giá gần dải dưới (có thể tăng)
  - bb_position ≈ 1: Giá gần dải trên (có thể giảm)
  - bb_position ≈ 0.5: Giá ở giữa

#### 6. Volatility (Biến động)
- **Công thức**: Standard deviation của returns
  ```python
  volatility = std(returns, window=20)
  ```
- **Normalization**: Chia cho max volatility
- **Ý nghĩa**: Volatility cao = rủi ro cao

#### 7. Current Profit
- **Tính toán**:
  ```python
  if position == 1:  # Đang hold
      current_profit = (current_price - entry_price) / entry_price
  else:
      current_profit = 0.0
  ```
- **Mục đích**: Agent biết đang lời/lỗ bao nhiêu để quyết định bán

#### 8. Time Since Trade
- **Tính toán**:
  ```python
  if position == 1:
      time_norm = min(steps_since_entry / max_hold_steps, 1.0)
  else:
      time_norm = 0.0
  ```
- **Mục đích**: Tránh hold quá lâu, khuyến khích trading

### Cơ Sở Lựa Chọn State Space

**Tại sao chọn 8 features này?**

1. **Portfolio State (position)**: Cần thiết để biết có thể Buy/Sell không
2. **Price Action (trend, bb_position)**: Phản ánh xu hướng giá
3. **Momentum Indicators (rsi, macd)**: Đo sức mạnh xu hướng
4. **Volatility**: Đánh giá rủi ro
5. **Trade Tracking (current_profit, time_since_trade)**: Quản lý giao dịch hiện tại

**So sánh với các nghiên cứu khác**:
- Paper gốc dùng 7 features (không có time_since_trade)
- Chúng ta thêm `time_since_trade` để tránh hold quá lâu
- Các nghiên cứu khác dùng 5-15 features tùy theo độ phức tạp

---

## ⚡ Không Gian Hành Động (Action Space)

### Định Nghĩa Action Space

```python
A = {0, 1, 2}

# Mapping:
action_mapping = {
    0: "BUY",   # Mua coin
    1: "SELL",  # Bán coin
    2: "HOLD"   # Giữ nguyên
}
```

### Chi Tiết Từng Action

#### Action 0: BUY (Mua)

**Điều kiện thực thi**:
```python
if action == 0 and position == 0 and balance > 0:
    # Có thể mua nếu:
    # - Đang không giữ coin (position = 0)
    # - Còn tiền trong tài khoản (balance > 0)
```

**Cơ chế thực hiện**:
```python
# Position Sizing (Risk Management)
max_investment = balance * max_position_pct  # e.g., 50% balance

# Transaction Cost
effective_balance = investment * (1 - transaction_cost)  # 0.01% fee

# Execute Buy
holdings = effective_balance / current_price
balance = balance - investment
position = 1
entry_price = current_price  # Track entry price
```

**Ví dụ**:
```
Balance: $10,000
Action: BUY
Price: $50,000/BTC
Max Position: 50% → $5,000
Fee: 0.01% → $0.50

Result:
- Holdings: 0.0999 BTC (= $4,999.50 / $50,000)
- Balance: $5,000 (còn lại 50%)
- Position: 1 (holding)
```

#### Action 1: SELL (Bán)

**Điều kiện thực thi**:
```python
if action == 1 and position == 1 and holdings > 0:
    # Có thể bán nếu:
    # - Đang giữ coin (position = 1)
    # - Có coin để bán (holdings > 0)
```

**Cơ chế thực hiện**:
```python
# Calculate sell value
sell_value = holdings * current_price

# Transaction Cost
balance = sell_value * (1 - transaction_cost)  # After 0.01% fee

# Execute Sell
holdings = 0
position = 0
entry_price = 0  # Reset entry price
```

**Ví dụ**:
```
Holdings: 0.1 BTC
Price: $55,000/BTC
Action: SELL

Calculation:
- Sell Value: 0.1 × $55,000 = $5,500
- Fee: 0.01% → $0.55
- Net Balance: $5,499.45

Result:
- Holdings: 0 BTC
- Balance: $5,499.45
- Position: 0 (cash)
- Profit: $499.45 (5% gain từ $5,000 investment)
```

#### Action 2: HOLD (Giữ nguyên)

**Điều kiện**: Luôn có thể hold

**Cơ chế**:
```python
# Không thay đổi balance hoặc holdings
# Chỉ cập nhật:
consecutive_holds += 1  # Đếm số lần hold liên tiếp
```

**Ý nghĩa**:
- Hold khi chưa có tín hiệu rõ ràng
- Hold khi đang chờ xu hướng rõ hơn
- Tránh over-trading (giao dịch quá nhiều)

### Cơ Sở Định Nghĩa Action Space

**Tại sao chỉ có 3 actions?**

1. **Đơn giản và hiệu quả**: 
   - Đủ để mô tả mọi tình huống trading
   - Phù hợp với paper gốc
   
2. **Không có "Short"**: 
   - Cryptocurrency market không hỗ trợ short dễ dàng như stocks
   - Giảm complexity của model

3. **Không có "Partial Buy/Sell"**:
   - Simplified version: All-in hoặc All-out
   - Có thể mở rộng bằng position sizing

**So sánh với các nghiên cứu**:

| Research | Action Space | Note |
|----------|-------------|------|
| Paper gốc | {Buy, Sell, Hold} | 3 actions |
| Deng et al. (2017) | {Buy, Sell, Hold} + position sizing | 5 actions |
| Theate & Ernst (2021) | Continuous action (%) | Complex |
| **Dự án này** | {Buy, Sell, Hold} | 3 actions, position sizing riêng |

### Cooldown Mechanism (V4+)

**Vấn đề**: Agent có thể over-trade (mua bán quá nhiều)

**Giải pháp**: Thêm cooldown giữa các trades
```python
trade_cooldown = 24  # 24 hours (cho 1h timeframe)
steps_since_last_trade = current_step - last_trade_step

if steps_since_last_trade < trade_cooldown and action in [BUY, SELL]:
    action = HOLD  # Force hold nếu còn cooldown
```

**Ý nghĩa**: 
- Tránh transaction cost tích lũy
- Mô phỏng real trading (cần thời gian observe market)

---

## 🎁 Hàm Thưởng và Phạt (Reward & Penalty Function)

### Tổng Quan Reward Function

Reward function là **trái tim** của Reinforcement Learning. Nó định nghĩa "goal" của agent.

**Mục tiêu**: 
- Maximize profit (tối đa hóa lợi nhuận)
- Follow trend (theo xu hướng đúng)
- Minimize risk (giảm rủi ro)

### Công Thức Reward Tổng Quát

```python
reward = portfolio_change_reward 
         + trend_bonus_reward 
         + profit_reward 
         - transaction_cost_penalty
         - hold_penalty
         + final_bonus (if episode ends)
```

### Chi Tiết Từng Thành Phần Reward

#### 1. Portfolio Change Reward (Chính)

**Công thức**:
```python
portfolio_value = balance + holdings * current_price
next_portfolio_value = balance + holdings * next_price

if portfolio_value > 0:
    portfolio_change = (next_portfolio_value - portfolio_value) / initial_balance
    reward += portfolio_change * 5.0  # Scaling factor
```

**Giải thích**:
- Reward dương nếu portfolio tăng giá trị
- Reward âm nếu portfolio giảm giá trị
- Scale ×5 để reward có magnitude phù hợp

**Ví dụ**:
```
Initial Balance: $10,000
Portfolio Value: $10,500 → $10,600

portfolio_change = ($10,600 - $10,500) / $10,000 = 0.01 (1%)
reward += 0.01 × 5.0 = +0.05
```

#### 2. Trend Bonus Reward

**Cơ sở từ paper**: 
> "Phần thưởng nếu hành động thực hiện đúng chiều xu hướng sẽ được cho là luôn dương. Ngược lại, nếu không đúng theo xu hướng thì hàm phần thưởng sẽ âm."

**Logic**:
```python
current_trend = data['trend']  # -1: giảm, 0: ngang, 1: tăng

# Check action đúng xu hướng
action_correct_trend = False

if action == BUY:
    action_correct_trend = (current_trend > 0)  # Mua khi tăng
elif action == SELL:
    action_correct_trend = (current_trend < 0)  # Bán khi giảm
elif action == HOLD:
    action_correct_trend = True  # Hold luôn OK

# Apply reward
if action_correct_trend:
    reward += 0.1  # Bonus cho đúng xu hướng
else:
    reward -= 0.1  # Penalty cho sai xu hướng
```

**Ví dụ**:
```
Scenario 1: Trend tăng, Action = BUY
→ reward += 0.1 ✅

Scenario 2: Trend tăng, Action = SELL
→ reward -= 0.1 ❌

Scenario 3: Trend giảm, Action = SELL
→ reward += 0.1 ✅
```

#### 3. Profit/Loss Reward

**Công thức**:
```python
# Tính profit nếu đang hold
if holdings > 0 and entry_price > 0:
    trade_profit_pct = (current_price - entry_price) / entry_price
    
    if trade_profit_pct > 0:
        reward += trade_profit_pct * 10.0  # Scale profit
    elif trade_profit_pct < 0:
        reward += trade_profit_pct * 10.0  # Penalty for loss (âm)
```

**Ví dụ**:
```
Entry Price: $50,000
Current Price: $52,000
Profit: 4%

reward += 0.04 × 10.0 = +0.4
```

#### 4. Transaction Cost Penalty

**Công thức**:
```python
transaction_cost = 0.0001  # 0.01% fee

if action in [BUY, SELL]:
    # Penalty đã được tính trong portfolio change
    # (balance giảm do fee khi buy/sell)
```

**Ý nghĩa**: 
- Phản ánh real-world trading cost
- Khuyến khích agent trade ít lần nhưng hiệu quả

#### 5. Hold Penalty (Deprecated trong V6+)

**Lý do loại bỏ**: 
- Paper gốc không có hold penalty
- Hold là action hợp lệ, không nên penalty
- Đã thay bằng cooldown mechanism

#### 6. Final Bonus (Episode End)

**Công thức**:
```python
if done:  # Episode kết thúc
    final_profit_pct = (portfolio_value - initial_balance) / initial_balance
    
    if final_profit_pct > 0:
        reward += final_profit_pct * 500.0  # HUGE bonus!
```

**Ý nghĩa**:
- Khuyến khích agent tối ưu toàn bộ episode
- Reward lớn (500x) để compensate cho delayed reward

**Ví dụ**:
```
Initial: $10,000
Final: $12,000
Profit: 20%

final_reward = 0.20 × 500.0 = +100.0 (rất lớn!)
```

### Bảng Tóm Tắt Reward Components

| Component | Formula | Scaling | Purpose |
|-----------|---------|---------|---------|
| Portfolio Change | (portfolio_t+1 - portfolio_t) / initial | ×5 | Chính: Track portfolio growth |
| Trend Bonus | +0.1 if correct, -0.1 if wrong | ±0.1 | Follow market trend |
| Profit Reward | (price - entry_price) / entry_price | ×10 | Reward profitable trades |
| Transaction Cost | Built into portfolio change | - | Real-world cost |
| Final Bonus | (final - initial) / initial | ×500 | Huge episode reward |

### Hàm Phạt (Penalty) Đâu?

**Câu hỏi quan trọng**: "Còn hàm phạt đâu?"

**Trả lời**:

1. **Penalty ngầm trong Portfolio Change**:
   ```python
   # Nếu portfolio giảm → reward âm (= penalty)
   if portfolio_value giảm:
       reward = negative value  # Đây chính là penalty!
   ```

2. **Trend Penalty**:
   ```python
   if action sai xu hướng:
       reward -= 0.1  # Explicit penalty
   ```

3. **Transaction Cost**:
   ```python
   # Fee 0.01% làm giảm portfolio → implicit penalty
   ```

4. **No Stop-Loss Penalty** (V6+):
   - Paper gốc không có stop-loss
   - Agent tự học khi nào nên cut loss

**Kết luận**: 
- Penalty không phải là function riêng
- Penalty = Negative reward trong portfolio change
- Penalty = Explicit deduction cho wrong action

### So Sánh Với Các Nghiên Cứu Khác

| Research | Reward Function | Penalty |
|----------|----------------|---------|
| **Paper gốc** | Portfolio change + Trend bonus | Implicit (negative reward) |
| Moody et al. (1998) | Sharpe ratio | Risk-adjusted |
| Deng et al. (2017) | Portfolio return | No explicit penalty |
| **Dự án này** | Portfolio + Trend + Profit + Final bonus | Implicit + Transaction cost |

---

## 🧠 Thuật Toán Q-Learning

### Giới Thiệu Q-Learning

Q-Learning là thuật toán **model-free** và **off-policy** trong Reinforcement Learning.

**Mục tiêu**: Học được **Q-function** (action-value function):

```
Q(s, a) = Expected cumulative reward khi thực hiện action a ở state s
```

### Bellman Equation

**Công thức cập nhật Q-value**:

```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
                        └────────┬────────┘  └──┬──┘
                           TD Target        Old Value
                        
TD Error = r + γ max Q(s', a') - Q(s, a)
```

**Giải thích**:
- **α**: Learning rate (0 < α ≤ 1)
  - α cao: Học nhanh nhưng không ổn định
  - α thấp: Học chậm nhưng ổn định
  
- **γ**: Discount factor (0 ≤ γ < 1)
  - γ ≈ 0: Chỉ quan tâm immediate reward
  - γ ≈ 1: Quan tâm long-term reward
  
- **r**: Immediate reward từ action
- **max Q(s', a')**: Best Q-value ở state tiếp theo

### Tabular Q-Learning

**Cách hoạt động**:

1. **Q-Table**: Dictionary lưu Q-values
   ```python
   Q = defaultdict(lambda: np.zeros(n_actions))
   # Q[(state_1, state_2, ..., state_8)] = [Q(s,BUY), Q(s,SELL), Q(s,HOLD)]
   ```

2. **Discretization**: Convert continuous state → discrete
   ```python
   def discretize_state(state):
       bins = np.linspace(0, 1, n_bins)  # e.g., 10 bins
       discrete = tuple(np.digitize(state, bins))
       return discrete
   ```

3. **Action Selection**: Epsilon-greedy
   ```python
   if random() < epsilon:
       action = random_action()  # Explore
   else:
       action = argmax(Q[state])  # Exploit
   ```

4. **Q-Update**:
   ```python
   state_key = discretize_state(state)
   next_state_key = discretize_state(next_state)
   
   td_target = reward + gamma * max(Q[next_state_key])
   td_error = td_target - Q[state_key][action]
   
   Q[state_key][action] += alpha * td_error
   ```

### Hyperparameters

```python
Q_LEARNING_PARAMS = {
    'alpha': 0.075,          # Learning rate
    'gamma': 0.95,           # Discount factor
    'epsilon': 1.0,          # Initial exploration rate
    'epsilon_decay': 0.9999, # Decay per episode
    'epsilon_min': 0.05,     # Minimum exploration
    'n_bins': 10,            # Discretization bins
}
```

**Giải thích lựa chọn**:
- **alpha = 0.075**: Middle ground giữa fast learning và stability
- **gamma = 0.95**: Ưu tiên short-term profit hơn (phù hợp với volatile market)
- **epsilon decay = 0.9999**: Decay chậm để explore đủ (5000 episodes)
- **epsilon_min = 0.05**: Giữ 5% exploration để adapt với market changes

---

## 🚀 Deep Q-Network (DQN)

### Tại Sao Cần DQN?

**Vấn đề của Tabular Q-Learning**:
- State space quá lớn: 10^8 states với 8 dimensions × 10 bins
- Q-table cần quá nhiều memory
- Không generalize được cho unseen states

**Giải pháp**: Deep Q-Network
- Neural network approximates Q-function
- Input: State vector (8 dims)
- Output: Q-values for 3 actions

### Kiến Trúc Neural Network

```
Input Layer (8 neurons - state features)
    │
    ▼
Hidden Layer 1 (256 neurons)
    │ ReLU activation
    ▼
Hidden Layer 2 (128 neurons)
    │ ReLU activation
    ▼
Hidden Layer 3 (64 neurons)
    │ ReLU activation
    ▼
Output Layer (3 neurons - Q-values)
    │ No activation (linear)
    ▼
[Q(s,BUY), Q(s,SELL), Q(s,HOLD)]
```

**Code**:
```python
class QNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # No activation - Q-values
        )
    
    def forward(self, x):
        return self.network(x)
```

### Experience Replay Buffer

**Vấn đề**: Sequential data có correlation cao → unstable learning

**Giải pháp**: Experience Replay
```python
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=32):
        return random.sample(self.buffer, batch_size)
```

**Lợi ích**:
- Break correlation giữa consecutive samples
- Reuse experiences nhiều lần
- Stable training

### Target Network

**Vấn đề**: Q-network update target cũng thay đổi → moving target

**Giải pháp**: Separate target network
```python
# Main network: Update mỗi step
Q_network.parameters()

# Target network: Update mỗi 10 episodes (hoặc C steps)
Q_target.load_state_dict(Q_network.state_dict())
```

**Q-Update với Target Network**:
```python
# Compute target Q-value sử dụng target network
with torch.no_grad():
    target_Q = reward + gamma * Q_target(next_state).max()

# Compute current Q-value sử dụng main network
current_Q = Q_network(state)[action]

# Loss
loss = F.mse_loss(current_Q, target_Q)

# Backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Training Algorithm

```python
for episode in range(num_episodes):
    state = env.reset()
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    for step in range(max_steps):
        # 1. Select action (epsilon-greedy)
        if random() < epsilon:
            action = random_action()
        else:
            with torch.no_grad():
                Q_values = Q_network(state)
                action = argmax(Q_values)
        
        # 2. Execute action
        next_state, reward, done, info = env.step(action)
        
        # 3. Store experience
        replay_buffer.push(state, action, reward, next_state, done)
        
        # 4. Train from replay buffer
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = compute_loss(batch, Q_network, Q_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 5. Update target network (every C episodes)
        if episode % target_update_freq == 0:
            Q_target.load_state_dict(Q_network.state_dict())
        
        if done:
            break
        
        state = next_state
```

### Hyperparameters DQN

```python
DQN_PARAMS = {
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'replay_buffer_size': 100000,
    'batch_size': 32,
    'target_update_freq': 10,  # episodes
}
```

---

## 📊 Technical Indicators

### RSI (Relative Strength Index)

**Công thức**:
```
RS = Average Gain / Average Loss (14 periods)
RSI = 100 - (100 / (1 + RS))
```

**Code**:
```python
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

**Ý nghĩa**:
- RSI > 70: Overbought (có thể giảm)
- RSI < 30: Oversold (có thể tăng)

### MACD (Moving Average Convergence Divergence)

**Công thức**:
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(MACD, 9)
MACD Histogram = MACD Line - Signal Line
```

**Ý nghĩa**:
- Histogram > 0: Bullish (xu hướng tăng)
- Histogram < 0: Bearish (xu hướng giảm)
- Crossover: Tín hiệu mua/bán

### Bollinger Bands

**Công thức**:
```
Middle Band = SMA(20)
Upper Band = SMA(20) + 2 × STD(20)
Lower Band = SMA(20) - 2 × STD(20)
```

**Ý nghĩa**:
- Price > Upper: Overbought
- Price < Lower: Oversold
- Bands expand: High volatility
- Bands contract: Low volatility

---

## 🛡️ Risk Management

### Position Sizing

```python
max_position_pct = 0.5  # Chỉ invest 50% balance
max_investment = balance * max_position_pct
```

**Lý do**: Giảm risk, giữ liquidity

### Transaction Cost

```python
transaction_cost = 0.0001  # 0.01% fee
effective_balance = investment * (1 - transaction_cost)
```

**Ý nghĩa**: Phản ánh real-world trading cost

### Maximum Drawdown Tracking

```python
# Update peak
if portfolio_value > peak_portfolio:
    peak_portfolio = portfolio_value

# Calculate MDD
mdd = (peak_portfolio - portfolio_value) / peak_portfolio
```

**Mục đích**: Monitor risk, không dùng để stop (theo paper)

---

## 📚 Tổng Kết

Phương pháp này kết hợp:
1. **MDP framework**: Formalize trading problem
2. **Q-Learning/DQN**: Learn optimal policy
3. **Technical Indicators**: Capture market information
4. **Risk Management**: Control downside risk

→ Tạo nên hệ thống trading tự động hoàn chỉnh.

**Xem [EXPERIMENTS.md](EXPERIMENTS.md) để biết chi tiết các thực nghiệm và [RESULTS.md](RESULTS.md) để xem kết quả.**
