# Hệ Thống Giao Dịch Tiền Điện Tử Tự Động với Q-Learning / DQN

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/Flask-2.x-green?logo=flask" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
  <img src="https://img.shields.io/badge/GPU-CUDA%20Optional-76B900?logo=nvidia" />
</p>

> **Dự án nghiên cứu** xây dựng hệ thống giao dịch tự động cho thị trường tiền điện tử sử dụng **Reinforcement Learning (Q-Learning & Deep Q-Network)**. Hệ thống học cách tối ưu hóa chiến lược mua/bán/giữ trên 5 loại tiền điện tử (BTC, ETH, BNB, SOL, ADA) từ 173,683 nến dữ liệu theo giờ (2016–2025).

---

## Mục Lục

1. [Tổng Quan Dự Án](#tổng-quan-dự-án)
2. [Kết Quả Nổi Bật](#kết-quả-nổi-bật)
3. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
4. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
5. [Cài Đặt và Yêu Cầu](#cài-đặt-và-yêu-cầu)
6. [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
7. [Web Interface](#web-interface)
8. [Tài Liệu Liên Quan](#tài-liệu-liên-quan)
9. [Disclaimer](#disclaimer)


---

## Tổng Quan Dự Án

Dự án xây dựng **hệ thống giao dịch tự động** (automated trading system) cho thị trường tiền điện tử sử dụng **Reinforcement Learning** — mô hình hóa bài toán dưới dạng **Markov Decision Process (MDP)** với agent tự học chiến lược tối ưu thông qua tương tác với môi trường thị trường.

### Đặc Điểm Chính

| Thành phần | Chi tiết |
|---|---|
| **Thuật toán** | Tabular Q-Learning, Deep Q-Network (DQN), DQN + PSO + LSTM |
| **Thị trường** | 5 cryptocurrency: BTC, ETH, BNB, SOL, ADA |
| **Khung thời gian** | Dữ liệu theo giờ (1h timeframe) |
| **Dữ liệu** | 173,683 mẫu, năm 2016–2025 |
| **Không gian trạng thái** | 8 chiều: vị thế, RSI, MACD, xu hướng, Bollinger Bands, ATR, win rate, loss streak |
| **Không gian hành động** | 3 hành động: Buy / Sell / Hold |
| **Chỉ báo kỹ thuật** | RSI, MACD, Bollinger Bands, ADX, SMA/EMA |
| **GPU Support** | CUDA (optional) để tăng tốc training |

---

## Kết Quả Nổi Bật

### Mô Hình Tốt Nhất — Version 6.1 (No Stop-Loss)

| Metric | Q-Learning | Buy & Hold |
|---|---|---|
| **Total Return** | **+38%** | +15% |
| **Annualized Return** | **+52%** | +18% |
| **Sharpe Ratio** | **1.42** | 0.61 |
| **Win Rate** | **64%** | — |
| **Max Drawdown** | 87% | 62% |
| **Số giao dịch** | 51 | — |

> Vượt trội hơn chiến lược Buy & Hold **+23 điểm phần trăm** lợi nhuận.  
> Sharpe Ratio 1.42 — mức xuất sắc theo chuẩn quản lý quỹ.  
> Win Rate 64% — cao hơn ngưỡng chuyên nghiệp 60%.

---

## Kiến Trúc Hệ Thống

```
+-----------------------------------------------------------+
|                     DU LIEU DAU VAO                       |
|         Binance API / Kaggle / yfinance (1h OHLCV)        |
+----------------------+------------------------------------+
                       |
                       v
+-----------------------------------------------------------+
|               TIEN XU LY & CHI BAO KY THUAT              |
|         RSI · MACD · Bollinger Bands · ADX · ATR          |
+----------------------+------------------------------------+
                       |
                       v
+-----------------------------------------------------------+
|             MOI TRUONG GIAO DICH (MDP)                    |
|  State (8D) -> Agent -> Action (Buy/Sell/Hold) -> Reward   |
|                 ^__________________________|              |
+----------------------+------------------------------------+
                       |
           +-----------+-----------+
           v                       v
+-----------------+     +----------------------+
|  Tabular        |     |  Deep Q-Network (DQN) |
|  Q-Learning     |     |  + PSO + LSTM         |
+--------+--------+     +----------+-----------+
         +----------+----------+
                    v
+-----------------------------------------------------------+
|               WEB DASHBOARD (Flask)                       |
|        Du doan real-time · Chi bao · Confidence           |
+-----------------------------------------------------------+
```

---

## Cấu Trúc Dự Án

```
Crypto_Trading_DQN/
|
+-- README.md                        # File nay -- Tong quan du an
+-- METHODOLOGY.md                   # Chi tiet phuong phap nghien cuu (MDP, thuat toan)
+-- EXPERIMENTS.md                   # Cac thuc nghiem va phan tich ket qua
+-- RESULTS.md                       # Bang metrics va so sanh hieu nang
+-- REFERENCES.md                    # Tai lieu tham khao hoc thuat
|
+-- requirements.txt                 # Dependencies CPU
+-- requirements_gpu.txt             # Dependencies GPU (CUDA)
|
+-- src/                            # Source code chinh
|   +-- config/
|   |   +-- config.py               # Cau hinh he thong (hyperparameters, paths)
|   +-- data/
|   |   +-- binance_data.py         # Tai du lieu tu Binance API
|   |   +-- kaggle_data.py          # Tai du lieu tu Kaggle
|   |   +-- multi_coin_loader.py    # Load & xu ly multi-coin data
|   +-- models/
|   |   +-- mdp_trading.py          # Moi truong MDP giao dich
|   |   +-- q_learning_gpu.py       # Tabular Q-Learning agent
|   |   +-- dqn_agent.py            # Deep Q-Network agent
|   |   +-- dqn_network.py          # Kien truc neural network
|   |   +-- dqn_pso_lstm_trading.py # Advanced: DQN + PSO + LSTM
|   |   +-- replay_buffer.py        # Experience replay buffer
|   |   +-- metrics.py              # Tinh toan metrics (MDD, Sharpe, Sortino...)
|   +-- utils/
|   |   +-- indicators.py           # Chi bao ky thuat (RSI, MACD, BB, ADX)
|   |   +-- checkpoint.py           # Luu/tai model checkpoint
|   +-- visualization/
|       +-- charts.py               # Cong cu ve bieu do
|       +-- trading_chart.py        # Bieu do giao dich chuyen biet
|
+-- train_dqn_pso_lstm.py           # Script training chinh (DQN + PSO + LSTM)
+-- validate_model.py                # Validate voi validation set
+-- validate_realtime.py             # Validate voi du lieu real-time
+-- visualize_training.py            # Visualize qua trinh training
|
+-- data/
|   +-- raw/
|   |   +-- multi_coin_1h.csv       # Du lieu 5 coins da gop (OHLCV)
|   +-- processed/                  # Du lieu da xu ly & model artifacts
|
+-- checkpoints/                    # Model checkpoints tu training
+-- results/                        # Ket qua, bieu do, reports
|
+-- web/                            # Web interface (Flask)
|   +-- app.py                      # Flask application
|   +-- templates/                  # HTML templates
|   +-- static/                     # CSS, JS, assets
|   +-- README_WEBAPP.md            # Huong dan web app
|
+-- tests/                          # Unit tests
```

---

## Cài Đặt và Yêu Cầu

### Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu |
|---|---|
| **Python** | 3.8 trở lên |
| **RAM** | Tối thiểu 8GB (khuyến nghị 16GB+) |
| **GPU** | NVIDIA CUDA-compatible (tùy chọn, để tăng tốc training) |
| **Dung lượng** | ~5GB cho data và models |

### 1. Clone Repository

```bash
git clone https://github.com/TiesnHunk/Crypto_Trading_DQN.git
cd Crypto_Trading_DQN
```

### 2. Cài Đặt Dependencies

```bash
# CPU (pho thong)
pip install -r requirements.txt

# GPU (CUDA -- training nhanh hon)
pip install -r requirements_gpu.txt
```

### 3. Cấu Hình

Chỉnh sửa `src/config/config.py` theo nhu cầu:

```python
# GPU Configuration
USE_GPU = True  # Dat False neu khong co GPU
DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

# Von ban dau
INITIAL_BALANCE = 10000.0

# Q-Learning Parameters
Q_LEARNING_PARAMS = {
    'state_dim': 8,          # So chieu state space
    'n_actions': 3,          # Buy=0, Sell=1, Hold=2
    'alpha': 0.075,          # Learning rate
    'gamma': 0.95,           # Discount factor
    'epsilon': 1.0,          # Exploration rate ban dau
    'epsilon_decay': 0.9999,
    'epsilon_min': 0.05,
}

TRAINING_EPISODES = 5000
```

---

## Hướng Dẫn Sử Dụng

### Bước 1 — Tải Dữ Liệu

```bash
# Tai tu Binance API
python src/data/binance_data.py

# Hoac tai tu Kaggle
python src/data/kaggle_data.py

# Chuan bi multi-coin dataset
python src/data/prepare_multi_coin_data.py
```

### Bước 2 — Training Model

#### Q-Learning (Tabular)
```bash
python src/main_multi_coin.py
```

#### Deep Q-Network (DQN)
```bash
python src/main_multi_coin_dqn.py
```

#### DQN + PSO + LSTM (nâng cao — có tối ưu hóa hyperparameter)
```bash
# Training mot coin voi PSO
python train_dqn_pso_lstm.py --coin BTC --episodes 100 --particles 5

# Resume tu checkpoint
python resume_training.py
```

> Checkpoint được lưu tự động vào `checkpoints/`, best model vào `data/processed/`

### Bước 3 — Đánh Giá và Validation

```bash
# Validate voi validation set
python validate_model.py

# Validate real-time voi du lieu moi nhat
python validate_realtime.py

# So sanh cac mo hinh
python compare_all_models.py
```

### Bước 4 — Visualization

```bash
# Bieu do qua trinh training
python visualize_training.py

# Ket qua theo chuan paper
python visualize_paper_style.py

# Theo doi training real-time
python watch_training.py
```

---

## Web Interface

Dashboard dự đoán real-time sử dụng DQN model đã training.

```bash
cd web
python app.py
# Truy cap: http://localhost:5000
```

### Tính Năng

- **Dữ liệu real-time** từ Binance API
- **Dự đoán AI**: BUY / SELL / HOLD với điểm tin cậy (%)
- **Hiển thị chỉ báo**: RSI, MACD, Bollinger Bands, ADX
- **Q-Values**: Giá trị kỳ vọng từng hành động

### API Endpoints

| Endpoint | Mô tả |
|---|---|
| `GET /api/predict` | Lấy dự đoán giao dịch hiện tại |
| `GET /api/status` | Kiểm tra trạng thái hệ thống |

**Ví dụ response `/api/predict`:**

```json
{
  "prediction": {
    "action_name": "MUA",
    "confidence": 85.3,
    "q_values": { "buy": 2.45, "sell": -0.82, "hold": -1.23 }
  },
  "current_price": 65432.50,
  "indicators": { "rsi": 58.3, "macd": 234.5, "adx": 28.7 }
}
```

> Xem thêm: [`web/README_WEBAPP.md`](web/README_WEBAPP.md)

---

## Tài Liệu Liên Quan

| File | Nội dung |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | Định nghĩa MDP, reward function, thuật toán Q-Learning & DQN, risk management |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Setup thực nghiệm, các phiên bản cải tiến, phân tích từng bước, lessons learned |
| [RESULTS.md](RESULTS.md) | Bảng metrics đầy đủ, biểu đồ performance, so sánh các phương pháp |
| [REFERENCES.md](REFERENCES.md) | Papers nghiên cứu liên quan, so sánh với công trình khác |
| [TRAIN_DQN_PSO_LSTM_README.md](TRAIN_DQN_PSO_LSTM_README.md) | Hướng dẫn training script nâng cao (PSO + LSTM) |
| [web/README_WEBAPP.md](web/README_WEBAPP.md) | Hướng dẫn cài đặt và sử dụng web interface |

---

## Changelog

| Phiên bản | Thay đổi chính |
|---|---|
| **V6.1** (Latest) | Bỏ stop-loss (theo paper gốc), cải thiện MDD tracking, reward function chuẩn |
| V6.0 | Thêm MDD tracking + penalty, annualized return, paper-compliant implementation |
| V5.0 | Multi-coin training (5 coins), GPU acceleration, experience replay buffer |
| V4.0 | Trading cooldown mechanism, improved risk management, state space 8D |

---

## Disclaimer

> **CANH BAO**: Dự án này **chỉ mang tính chất nghiên cứu và giáo dục**.

- **KHONG** sử dụng trực tiếp để giao dịch tiền thật
- Kết quả backtesting **không đảm bảo** hiệu quả trong tương lai
- Trading cryptocurrency có rủi ro cao, có thể **mất toàn bộ vốn**
- Luôn tham khảo ý kiến chuyên gia tài chính trước khi đầu tư

---

<p align="center">Made with Q-Learning x DQN x Flask · 2025</p>
