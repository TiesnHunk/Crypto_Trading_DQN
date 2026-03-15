# SO SÁNH DQN TRÊN 3 MARKET REGIMES

## Tóm Tắt

Nghiên cứu này đánh giá hiệu suất của Deep Q-Network (DQN) trên 3 loại thị trường khác nhau:
- **Bull Market**: Thị trường tăng giá mạnh
- **Bear Market**: Thị trường giảm giá mạnh  
- **Sideways Market**: Thị trường đi ngang

## 1. Các Ngày Test

| Market | Date | Price Change | Trend Avg | RSI Avg |
|--------|------|--------------|-----------|---------|
| BULL | 2021-02-08 | 19.75% | 0.417 | 70.13 |
| BEAR | 2020-03-12 | -39.34% | -1.000 | 22.12 |
| SIDEWAYS | 2025-02-07 | -0.00% | -0.083 | 52.12 |

## 2. Kết Quả Chi Tiết

### 2.1. Bảng So Sánh Tổng Quan

| Market   | Date       | Price Change   |   Trend Avg |   RSI Avg | Total Return   | Final Balance   |   Num Trades | Hold %   | Buy %   | Sell %   |    MAE | Accuracy   |       R² |
|:---------|:-----------|:---------------|------------:|----------:|:---------------|:----------------|-------------:|:---------|:--------|:---------|-------:|:-----------|---------:|
| BULL     | 2021-02-08 | 19.75%         |       0.417 |     70.13 | 9.87%          | $5000.00        |           22 | 8.3%     | 0.0%    | 91.7%    | 1.4583 | 12.50%     |  -7.2623 |
| BEAR     | 2020-03-12 | -39.34%        |      -1     |     22.12 | -19.67%        | $5000.00        |           17 | 29.2%    | 66.7%   | 4.2%     | 0.6667 | 37.50%     |   0.069  |
| SIDEWAYS | 2025-02-07 | -0.00%         |      -0.083 |     52.12 | -0.01%         | $5000.00        |           23 | 4.2%     | 0.0%    | 95.8%    | 1.75   | 12.50%     | -10.4545 |

### 2.2. Trading Performance


#### BULL Market (2021-02-08)

**Market Conditions:**
- Price change: 19.75%
- Trend average: 0.417
- RSI average: 70.13

**DQN Performance:**
- Total Return: 9.87%
- Final Balance: $5000.00
- Number of Trades: 22

**Action Distribution:**
- Hold: 2 (8.3%)
- Buy: 0 (0.0%)
- Sell: 22 (91.7%)

**Metrics:**
- MAE: 1.4583
- Accuracy: 12.50%
- R² Score: -7.2623


#### BEAR Market (2020-03-12)

**Market Conditions:**
- Price change: -39.34%
- Trend average: -1.000
- RSI average: 22.12

**DQN Performance:**
- Total Return: -19.67%
- Final Balance: $5000.00
- Number of Trades: 17

**Action Distribution:**
- Hold: 7 (29.2%)
- Buy: 16 (66.7%)
- Sell: 1 (4.2%)

**Metrics:**
- MAE: 0.6667
- Accuracy: 37.50%
- R² Score: 0.0690


#### SIDEWAYS Market (2025-02-07)

**Market Conditions:**
- Price change: -0.00%
- Trend average: -0.083
- RSI average: 52.12

**DQN Performance:**
- Total Return: -0.01%
- Final Balance: $5000.00
- Number of Trades: 23

**Action Distribution:**
- Hold: 1 (4.2%)
- Buy: 0 (0.0%)
- Sell: 23 (95.8%)

**Metrics:**
- MAE: 1.7500
- Accuracy: 12.50%
- R² Score: -10.4545


## 3. Phân Tích và Nhận Xét

### 3.1. Bull Market Performance


**Kết quả:**
- DQN return: 9.87%
- Market return: 19.75%
- Buy actions: 0.0%

**Nhận xét:**
✗ DQN không tận dụng tốt bull market, quá ít lệnh Buy

### 3.2. Bear Market Performance


**Kết quả:**
- DQN return: -19.67%
- Market return: -39.34%
- Sell actions: 4.2%

**Nhận xét:**
✗ DQN không bảo vệ vốn tốt trong bear market

### 3.3. Sideways Market Performance


**Kết quả:**
- DQN return: -0.01%
- Market return: -0.00%
- Trading frequency: 95.8%

**Nhận xét:**
✓ DQN thực hiện range trading tốt trong sideways market

## 4. Kết Luận Chung

### 4.1. Market-Adaptive Behavior

DQN cho thấy khả năng thích nghi với các market regime:

**Performance ranking:**
1. BULL: 9.87% return
2. SIDEWAYS: -0.01% return
3. BEAR: -19.67% return

### 4.2. Limitations

1. **Training Bias**: DQN có thể bị bias theo market regime chủ đạo trong training data
2. **Short-term Testing**: Chỉ test 24h/regime, cần test dài hạn hơn
3. **Reward Function**: Có thể cần điều chỉnh reward cho từng market type

### 4.3. Khuyến Nghị

1. **Adaptive Reward**: Điều chỉnh reward function theo market regime
2. **Ensemble Models**: Kết hợp nhiều models cho từng market type
3. **Market Detection**: Thêm module phát hiện market regime tự động
4. **Long-term Testing**: Test trên periods dài hơn (7-30 days)

---

**Ngày tạo**: 2025-12-15  
**Mô hình**: DQN (Episode 1831, Best Profit $3.8M)  
**Checkpoint**: src/checkpoints_dqn/checkpoint_best.pkl
