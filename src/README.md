# 📂 SRC Folder

Source code chính của project.

## 📁 Structure

```
src/
├── config/              # Configuration files
│   ├── config.py       # Main config (GPU, training, data period)
│   └── config_old.py   # Old config (backup)
├── data/               # Data fetching modules
│   ├── binance_data.py # Binance API integration
│   └── kaggle_data.py  # Kaggle data loader
├── models/             # ML models
│   ├── q_learning_gpu.py        # Q-Learning với GPU
│   ├── enhanced_training_gpu.py # Enhanced training
│   ├── mdp_trading.py           # MDP environment
│   ├── trend_trading.py         # Trend analysis
│   ├── enhanced_training.py     # Old training
│   ├── q_learning.py            # Old Q-Learning
│   └── metrics.py               # Performance metrics
├── utils/              # Utility functions
│   └── indicators.py   # Technical indicators
├── visualization/      # Plotting và charts
│   ├── charts.py       # Extended charts (GPU version)
│   ├── binance_chart.py # Binance charts
│   └── bitcoin_history.py # Historical charts
├── main_gpu.py         # 🚀 MAIN SCRIPT (GPU version)
├── main_complete.py    # Complete trading system (old)
├── main_enhanced.py    # Enhanced version (old)
├── main_historical.py  # Historical analysis
└── run_test.py         # Testing script
```

## 🚀 Main Scripts

### 1. main_gpu.py (RECOMMENDED)
**New GPU-accelerated version**
- Period: 2018-2025
- Episodes: 5000
- GPU support
- Extended charts

```bash
python src/main_gpu.py
```

### 2. main_complete.py (OLD)
Complete trading system với old config

### 3. main_enhanced.py (OLD)
Enhanced version với old config

### 4. main_historical.py
Historical data analysis

## 📦 Packages

### config/
Configuration management với GPU settings

### data/
Data fetching từ Binance và Kaggle

### models/
Machine learning models:
- Q-Learning (GPU + CPU versions)
- MDP Trading environment
- Training algorithms

### utils/
Utility functions:
- Technical indicators (RSI, MA, BB, etc.)

### visualization/
Visualization tools:
- Extended multi-panel charts
- Price analysis
- Performance metrics

## 🎯 Usage

Import modules:
```python
from src.config.config import DEVICE, TRAINING_EPISODES
from src.data.binance_data import BinanceDataFetcher
from src.models.q_learning_gpu import QLearningAgent
from src.visualization.charts import plot_extended_price_chart
```

Run main:
```bash
python src/main_gpu.py
```
