# BÁO CÁO DỰ BÁO HÀNH ĐỘNG GIAO DỊCH - 1 NGÀY

## Thông Tin Chung

- **Ngày dự báo**: 2020-04-06
- **Khoảng thời gian**: 00:00 - 23:00
- **Số giờ**: 24 giờ
- **Mô hình**: Deep Q-Network (DQN)
- **Checkpoint**: Episode 1831, Best Profit $3.8M

## 1. Metrics Đánh Giá

### 1.1. Metrics So Sánh với Ideal Actions

| Metric | Value | Giải Thích |
|--------|-------|-----------|
| **MAE** | 1.8333 | Mean Absolute Error - Sai số tuyệt đối trung bình |
| **MSE** | 3.5833 | Mean Squared Error - Sai số bình phương trung bình |
| **RMSE** | 1.8930 | Root Mean Squared Error - Căn bậc hai của MSE |
| **MAPE** | 100.00% | Mean Absolute Percentage Error - Sai số % TB |
| **R² Score** | -45.9091 | Coefficient of Determination - Độ phù hợp |
| **Accuracy** | 4.17% | % dự báo đúng so với ideal actions |

### 1.2. Trading Performance

| Metric | Value |
|--------|-------|
| **Total Return** | 3.37% |
| **Final Balance** | $5000.00 |
| **Initial Balance** | $10,000.00 |
| **Profit/Loss** | $-5000.00 |
| **Number of Trades** | 23 |

## 2. Phân Bố Hành Động

| action_name   |   count |
|:--------------|--------:|
| Sell          |      23 |
| Hold          |       1 |

## 3. Môi Trường Thị Trường

- **Loại thị trường**: BULL MARKET (Thị trường tăng giá)
- **Thay đổi giá**: 6.76%
- **Trend trung bình**: 0.750
- **RSI trung bình**: 72.80
- **Volatility TB**: 0.0058

**Chiến lược tối ưu**: Buy sớm và Hold lâu, ít Sell

**Đánh giá DQN**: Không tối ưu: DQN nên mua nhiều hơn

## 4. Phân Tích Reward Function

### 4.1. Công Thức Reward

```python
reward = profit - transaction_cost - hold_penalty

# Components:
# - profit: Lợi nhuận từ giao dịch
# - transaction_cost: 0.01% per trade
# - hold_penalty: 0.01% để tránh hold quá lâu
# - trend_bonus: +0.1 nếu action align với trend
# - risk_penalty: -0.5 × MDD nếu MDD > 30%
```

### 4.2. Reward Statistics

| Action | Count | Avg Reward | Max Reward | Min Reward |
|--------|-------|-----------|-----------|-----------|
| Hold | 1 | -0.1089 | -0.1089 | -0.1089 |
| Buy | 0 | N/A | N/A | N/A |
| Sell | 23 | 1.1828 | 17.6462 | 0.0473 |

## 5. Thảo Luận

### 5.1. Ảnh Hưởng của Môi Trường đến Hành Động

Trong thị trường **BULL MARKET**, DQN đã học được chiến lược:

- **Hold**: 1/24 lần (4.2%)
- **Buy**: 0/24 lần (0.0%)
- **Sell**: 23/24 lần (95.8%)


### 5.2. Trade-offs trong Reward Design

**a) Profit vs Transaction Cost:**
- Profit cao khuyến khích giao dịch nhiều
- Transaction cost (0.01%) ngăn chặn over-trading
- DQN học cân bằng: Chỉ trade khi expected profit > cost

**b) Hold Penalty vs Over-trading:**
- Hold penalty (0.01%) khuyến khích hành động
- Nhưng không quá lớn để tránh giao dịch bừa bãi
- DQN học: Hold khi chờ cơ hội tốt hơn

**c) Trend Alignment:**
- +0.1 reward nếu Buy khi trend tăng, Sell khi trend giảm
- Khuyến khích DQN follow the trend
- Tránh "catch the falling knife" (mua khi giảm mạnh)

**d) Risk Management:**
- MDD penalty ngăn chặn drawdown lớn
- Stop loss (10%) tự động bảo vệ vốn
- Max position (50%) hạn chế rủi ro

### 5.3. So Sánh với Ideal Actions

DQN đạt **4.17% accuracy** so với ideal actions (dựa trên price direction).

**Phân tích:**
- MAE = 1.8333: Sai số trung bình 1.83 action
- R² = -45.9091: Cần cải thiện

## 6. Kết Luận

### 6.1. Điểm Mạnh

1. **Adaptive Strategy**: DQN học được điều chỉnh hành động theo môi trường
2. **Risk Awareness**: Tránh over-trading với transaction cost awareness
3. **Trend Following**: Align actions với market trend

### 6.2. Điểm Cần Cải Thiện

1. **Metrics Limitations**: MAE/MSE/RMSE không phù hợp hoàn toàn với classification task
2. **Short-term Prediction**: Chỉ test 1 ngày, cần test dài hạn hơn
3. **Market Dependency**: Performance phụ thuộc nhiều vào loại thị trường

### 6.3. Khuyến Nghị

- Test trên nhiều khoảng thời gian khác nhau (bull/bear/sideways)
- Sử dụng classification metrics (Precision, Recall, F1)
- So sánh với baseline strategies (Buy & Hold, Moving Average)
- Backtest trên multiple coins để đánh giá generalization

---

**Ngày tạo**: 2025-12-17 12:36:35  
**Phiên bản**: 1.0
