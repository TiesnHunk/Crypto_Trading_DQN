# PHÂN TÍCH SO SÁNH MÔ HÌNH DỰ BÁO HÀNH ĐỘNG GIAO DỊCH

## 1. Tổng Quan

Nghiên cứu này so sánh hiệu suất của các mô hình học máy trong việc dự báo hành động giao dịch tiền điện tử (Bitcoin). Các mô hình được đánh giá bao gồm:

1. **LSTM (Long Short-Term Memory)** - Mô hình mạng nơ-ron hồi tiếp cơ bản
2. **PSO + LSTM** - Tối ưu hóa siêu tham số LSTM bằng Particle Swarm Optimization
3. **PPO + PSO + LSTM** - Kết hợp Proximal Policy Optimization với PSO và LSTM
4. **DQN (Deep Q-Network)** - Học tăng cường với Q-learning
5. **DQN + PSO + LSTM** - Kết hợp DQN với LSTM và tối ưu hóa PSO

## 2. Phương Pháp Đánh Giá

### 2.1. Không Gian Hành Động

Mỗi mô hình dự báo một trong ba hành động tại mỗi thời điểm:
- **Action 0: Hold (Giữ)** - Không thực hiện giao dịch
- **Action 1: Buy (Mua)** - Mua Bitcoin
- **Action 2: Sell (Bán)** - Bán Bitcoin

### 2.2. Metrics Đánh Giá

#### Metrics Chính:
1. **Total Return (%)**: Tỷ suất sinh lời tổng thể
   - Công thức: `(Final Balance - Initial Balance) / Initial Balance × 100`
   
2. **Final Balance ($)**: Số dư cuối cùng sau khi giao dịch
   - Initial Balance: $10,000
   
3. **Number of Trades**: Tổng số giao dịch thực hiện
   - Chỉ tính Buy và Sell, không tính Hold

#### Metrics Phụ:
- **Action Distribution**: Phân bố các hành động (Hold/Buy/Sell)
- **Portfolio Value Over Time**: Giá trị danh mục theo thời gian
- **Sharpe Ratio**: Tỷ lệ return/risk (nếu có đủ dữ liệu)

### 2.3. Môi Trường Giao Dịch

**Trading Environment (MDP - Markov Decision Process):**

**State Space (7 chiều):**
```python
state = [
    position,        # 0: không giữ, 1: đang giữ BTC
    rsi,            # Relative Strength Index (0-100)
    macd_hist,      # MACD histogram
    trend,          # Xu hướng: 1 (tăng), -1 (giảm), 0 (đi ngang)
    bb_position,    # Vị trí giá so với Bollinger Bands
    volatility,     # Độ biến động giá
    price_change    # % thay đổi giá
]
```

**Reward Function:**
```python
reward = profit - transaction_cost - hold_penalty
```

Trong đó:
- `profit`: Lợi nhuận từ giao dịch
- `transaction_cost`: Phí giao dịch (0.01% = 0.0001)
- `hold_penalty`: Penalty cho Hold khi không giao dịch (0.01%)

**Transaction Constraints:**
- **Stop Loss**: 10% - Tự động bán khi lỗ quá 10%
- **Max Position**: 50% - Chỉ dùng tối đa 50% số dư để mua
- **Max Loss**: 20% - Dừng episode nếu lỗ quá 20%
- **Cooldown**: 24h - Chờ 24 giờ giữa các giao dịch (cho dữ liệu 1h)

## 3. Phân Tích Môi Trường Thị Trường

### 3.1. Ảnh Hưởng của Xu Hướng Thị Trường

**Bullish Market (Thị trường tăng giá):**
- **Đặc điểm**: Giá tăng liên tục, RSI > 50, MACD > 0
- **Chiến lược tối ưu**: 
  - Buy sớm và Hold lâu
  - Sell ít, chỉ khi có tín hiệu đảo chiều mạnh
- **Models phù hợp**: 
  - DQN + PSO + LSTM: Học được pattern tăng trưởng
  - PPO + PSO + LSTM: Policy gradient thích nghi tốt

**Bearish Market (Thị trường giảm giá):**
- **Đặc điểm**: Giá giảm liên tục, RSI < 50, MACD < 0
- **Chiến lược tối ưu**:
  - Sell sớm để bảo toàn vốn
  - Buy ít hoặc không Buy
  - Hold cash để chờ đáy
- **Models phù hợp**:
  - DQN: Q-learning học được tránh rủi ro
  - Models có Stop Loss: Tự động cắt lỗ

**Sideways Market (Thị trường đi ngang):**
- **Đặc điểm**: Giá dao động trong khoảng hẹp
- **Chiến lược tối ưu**:
  - Buy ở đáy range, Sell ở đỉnh range
  - Trading tần suất cao hơn
- **Models phù hợp**:
  - PSO + LSTM: Tối ưu cho pattern lặp lại
  - DQN: Exploration-exploitation balance

### 3.2. Ảnh Hưởng của Phần Thưởng (Reward)

**Reward Shaping Impact:**

1. **Transaction Cost (0.01%)**:
   - Khuyến khích ít giao dịch
   - Models có xu hướng Hold nhiều hơn
   - Cần đánh đổi giữa trading frequency và profit

2. **Hold Penalty (0.01%)**:
   - Khuyến khích hành động (Buy/Sell)
   - Tránh tình trạng "lazy agent" - chỉ Hold
   - Balanced với transaction cost

3. **Profit Maximization**:
   - Reward chính từ profit
   - Khuyến khích timing tốt cho Buy/Sell
   - Long-term return > short-term gains

**Trade-offs:**
```
High Transaction Cost → More Hold → Fewer Trades → Lower Profit (if market trending)
Low Hold Penalty → More Hold → Miss Opportunities → Lower Return
```

### 3.3. Phân Tích Hành Động

**Buy Decision Factors:**
- RSI < 30 (oversold)
- MACD histogram chuyển dương
- Trend đảo chiều từ giảm sang tăng
- Price ở đường Bollinger Band dưới
- Volatility thấp (ít rủi ro)

**Sell Decision Factors:**
- RSI > 70 (overbought)
- MACD histogram chuyển âm
- Trend đảo chiều từ tăng sang giảm
- Price ở đường Bollinger Band trên
- Stop loss triggered (giảm > 10%)

**Hold Decision Factors:**
- Không có tín hiệu rõ ràng
- Đang trong cooldown period
- Volatility cao (rủi ro cao)
- Không đủ confidence để Buy/Sell

## 4. So Sánh Các Mô Hình

### 4.1. LSTM (Baseline)

**Ưu điểm:**
- Đơn giản, dễ implement
- Học được temporal patterns
- Xử lý tốt time series data

**Nhược điểm:**
- Không có optimization chiến lược giao dịch
- Chỉ dự báo giá, không tối ưu profit
- Overfit với training data

**Khi nào dùng:**
- Baseline comparison
- Quick prototyping
- Khi cần interpretability

### 4.2. PSO + LSTM

**Ưu điểm:**
- Tự động tìm hyperparameters tối ưu
- Cải thiện accuracy so với LSTM thuần
- Tránh local minima

**Nhược điểm:**
- Cần thời gian optimization
- Không đảm bảo global optimum
- Phụ thuộc vào PSO parameters

**Khi nào dùng:**
- Khi có thời gian để optimize
- Dataset lớn, nhiều features
- Cần improve LSTM performance

### 4.3. PPO + PSO + LSTM

**Ưu điểm:**
- Policy gradient method - học trực tiếp policy
- Sample efficient
- Stable training

**Nhược điểm:**
- Phức tạp nhất
- Cần nhiều training episodes
- Hyperparameter sensitive

**Khi nào dùng:**
- Continuous action space
- Khi cần adaptive policy
- High-frequency trading

### 4.4. DQN (Deep Q-Network)

**Ưu điểm:**
- Học trực tiếp từ reward
- Value-based method ổn định
- Experience replay giảm correlation

**Nhược điểm:**
- Discrete actions only
- Sample inefficient
- Overestimation bias

**Khi nào dùng:**
- Discrete action problems
- Khi có replay buffer
- Stable environment

### 4.5. DQN + PSO + LSTM (Hybrid)

**Ưu điểm:**
- Kết hợp điểm mạnh của cả 3 methods
- LSTM xử lý temporal data
- DQN optimize actions
- PSO tune hyperparameters

**Nhược điểm:**
- Rất phức tạp
- Training time lâu
- Khó debug

**Khi nào dùng:**
- Production deployment
- Khi cần best performance
- Đủ computational resources

## 5. Nhận Xét và Khuyến Nghị

### 5.1. Key Findings

1. **Market Condition is Critical:**
   - Performance khác nhau rất nhiều giữa bull và bear market
   - Models cần adaptive để handle market regime changes

2. **Reward Design Matters:**
   - Transaction cost và hold penalty cần balance
   - Reward shaping ảnh hưởng lớn đến learned policy

3. **Action Distribution Insights:**
   - Models tend to Hold > Buy > Sell
   - Profitable models có balance tốt giữa 3 actions

### 5.2. Khuyến Nghị Sử Dụng

**Cho Beginners:**
- Dùng DQN - Đơn giản, hiệu quả
- Focus vào understanding basics

**Cho Advanced Users:**
- DQN + PSO + LSTM cho best performance
- Tune hyperparameters carefully

**Cho Production:**
- Ensemble của nhiều models
- Risk management layers
- Regular retraining

### 5.3. Future Improvements

1. **Multi-Asset Trading**: Mở rộng sang nhiều coins
2. **Risk-Adjusted Returns**: Thêm Sharpe ratio vào reward
3. **Market Regime Detection**: Adaptive strategies
4. **Transfer Learning**: Share knowledge giữa các coins

## 6. Tài Liệu Tham Khảo

[Sẽ được thêm vào sau khi có kết quả cụ thể]

**Deep Reinforcement Learning:**
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv.

**Financial Applications:**
- Jiang et al. (2017). "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem." arXiv.
- Fischer & Krauss (2018). "Deep learning with long short-term memory networks for financial market predictions." EJOR.

**Optimization:**
- Kennedy & Eberhart (1995). "Particle swarm optimization." IEEE.

---

**Note**: Phân tích này sẽ được cập nhật với số liệu thực tế sau khi chạy xong script so sánh.
