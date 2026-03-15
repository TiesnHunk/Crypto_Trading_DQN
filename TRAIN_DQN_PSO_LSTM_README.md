# Training DQN + PSO + LSTM từ đầu

## 📋 Tổng Quan

Script `train_dqn_pso_lstm.py` cho phép training model **DQN + PSO + LSTM** từ đầu với các tính năng:

- ✅ **PSO Optimization**: Tối ưu hyperparameters tự động
- ✅ **Checkpoint System**: Lưu và resume training
- ✅ **Best Model Tracking**: Tự động lưu best model (train & test)
- ✅ **Training Visualization**: Vẽ đồ thị training progress
- ✅ **Evaluation**: Đánh giá trên test set

---

## 🚀 Cách Sử Dụng

### 1. Training Từ Đầu (Mặc Định)

```bash
python train_dqn_pso_lstm.py
```

**Mặc định:**
- Data: `data/raw/multi_coin_1h.csv`
- Coin: BTC
- PSO Particles: 5
- PSO Iterations: 10
- Training Episodes: 100
- Train/Test Split: 80/20
- Checkpoint dir: `checkpoints/dqn_pso_lstm/`

### 2. Training Với Custom Parameters

```bash
python train_dqn_pso_lstm.py \
    --coin ETH \
    --particles 8 \
    --iterations 15 \
    --episodes 200 \
    --train-split 0.85 \
    --checkpoint-dir checkpoints/my_dqn_pso \
    --save-interval 10 \
    --device cuda \
    --seed 42
```

### 3. Resume Training Từ Checkpoint

```bash
python train_dqn_pso_lstm.py \
    --resume checkpoints/dqn_pso_lstm/pso_optimization_results.json \
    --episodes 200
```

---

## 📝 Arguments Chi Tiết

### Data Arguments
- `--data`: Path đến file CSV (default: `data/raw/multi_coin_1h.csv`)
- `--coin`: Đồng coin để train (BTC, ETH, BNB, SOL, ADA) (default: `BTC`)

### PSO Arguments
- `--particles`: Số particles cho PSO (default: `5`)
  - Nhiều hơn = tìm kiếm tốt hơn nhưng chậm hơn
  - Khuyến nghị: 5-10
- `--iterations`: Số iterations của PSO (default: `10`)
  - Nhiều hơn = optimization tốt hơn nhưng chậm hơn
  - Khuyến nghị: 10-20

### Training Arguments
- `--episodes`: Số episodes training (default: `100`)
  - Khuyến nghị: 100-500 tùy data size
- `--train-split`: Tỷ lệ train/test split (default: `0.8`)
  - Range: 0.7 - 0.9

### Checkpoint Arguments
- `--checkpoint-dir`: Thư mục lưu checkpoints (default: `checkpoints/dqn_pso_lstm`)
- `--save-interval`: Lưu checkpoint mỗi N episodes (default: `20`)
- `--resume`: Path đến checkpoint để resume (default: `None`)

### Other Arguments
- `--device`: Device để train (`cuda` hoặc `cpu`) (default: auto-detect)
- `--seed`: Random seed (default: `42`)

---

## 📁 Cấu Trúc Checkpoints

Sau khi training, checkpoints được lưu trong folder:

```
checkpoints/dqn_pso_lstm/
├── pso_optimization_results.json       # PSO results + best params
├── pso_optimization_results_agent.pth  # Agent sau PSO
├── checkpoint_best.pth                 # Best model (train reward)
├── checkpoint_best_test.pth            # Best model (test reward)
├── checkpoint_latest.pth               # Latest model
├── checkpoint_final.pth                # Final model
├── checkpoint_ep20.pth                 # Checkpoint tại episode 20
├── checkpoint_ep40.pth                 # Checkpoint tại episode 40
├── ...
└── training_progress.png               # Đồ thị training
```

### Checkpoint Contents

Mỗi checkpoint (.pth) chứa:
- `policy_net_state_dict`: Weights của policy network
- `target_net_state_dict`: Weights của target network
- `optimizer_state_dict`: Optimizer state
- `epsilon`: Current epsilon value
- `steps`: Total training steps
- `episodes`: Total episodes
- `episode`: Current episode
- `best_reward`: Best reward achieved
- Hyperparameters (input_size, hidden_size, etc.)

---

## 📊 Output

### Console Output

```
================================================================================
🚀 DQN + PSO + LSTM TRAINING FROM SCRATCH
================================================================================

   Date: 2025-12-14 15:30:00
   Coin: BTC
   Device: cuda
   Seed: 42
   Checkpoint dir: checkpoints/dqn_pso_lstm

================================================================================
📥 LOADING DATA
================================================================================
   Total rows: 288,637
   BTC data: 68,542 rows
   Date range: 2018-04-17 04:00:00 to 2025-11-01 00:00:00

📊 Data Split:
   Train: 54,833 samples (2018-04-17 to 2024-10-15)
   Test: 13,709 samples (2024-10-15 to 2025-11-01)

================================================================================
🔬 PSO OPTIMIZATION PHASE
================================================================================
   Input size: 19 features
   Environment: 54,833 steps

   PSO Configuration:
   - Particles: 5
   - Iterations: 10
   - Inertia weight: 0.7
   - Cognitive/Social: 1.5/1.5

================================================================================
🚀 Starting PSO Optimization...
================================================================================

=== PSO Iteration 1/10 ===
Particle 1/5... Fitness: 125.34
Particle 2/5... Fitness: -45.67
...
Global Best Fitness: 125.34

...

================================================================================
✅ PSO OPTIMIZATION COMPLETE!
================================================================================

   Best Parameters:
      hidden_size: 128
      num_layers: 2
      sequence_length: 24
      learning_rate: 0.0015
      gamma: 0.98
      epsilon_decay: 0.995
      batch_size: 64
      target_update_frequency: 100

   Best Fitness: 245.78

   💾 Checkpoint saved: checkpoints/dqn_pso_lstm/pso_optimization_results_agent.pth
   💾 Results saved: checkpoints/dqn_pso_lstm/pso_optimization_results.json

================================================================================
🎯 TRAINING PHASE
================================================================================

   Training Configuration:
   - Episodes: 100
   - Sequence length: 24
   - Batch size: 64
   - Learning rate: 0.0015
   - Gamma: 0.98
   - Epsilon decay: 0.995

================================================================================
🚀 Starting Training...
================================================================================

Episode 1/100:
   Reward: 123.45 | Avg(10): 123.45 | Best: 123.45
   Loss: 0.0234 | Epsilon: 0.9950 | Steps: 854

Episode 10/100:
   Reward: 234.56 | Avg(10): 189.34 | Best: 245.67
   Loss: 0.0198 | Epsilon: 0.9512 | Steps: 912
   Test Reward: 198.76 | Best Test: 198.76

...

   💾 Checkpoint saved: checkpoints/dqn_pso_lstm/checkpoint_ep20.pth

...

================================================================================
✅ TRAINING COMPLETE!
================================================================================

   Best Training Reward: 456.78
   Best Test Reward: 389.45
   Final Epsilon: 0.6050

   💾 Checkpoint saved: checkpoints/dqn_pso_lstm/checkpoint_final.pth

   📊 Training plot saved: checkpoints/dqn_pso_lstm/training_progress.png

================================================================================
📊 FINAL EVALUATION
================================================================================

   Final Test Reward: 389.45
   Initial Balance: $10000.00
   Final Portfolio: $10389.45
   Total Return: 3.89%

================================================================================
✅ TRAINING COMPLETE!
================================================================================

   All checkpoints saved to: checkpoints/dqn_pso_lstm
   - checkpoint_best.pth (best training reward)
   - checkpoint_best_test.pth (best test reward)
   - checkpoint_final.pth (final model)
   - checkpoint_latest.pth (latest model)
```

---

## 🔄 Workflow

### Giai Đoạn 1: PSO Optimization (Auto)
1. Khởi tạo swarm của particles
2. Mỗi particle = 1 bộ hyperparameters
3. Train quick agent với mỗi bộ params
4. Evaluate fitness (reward)
5. Update particles theo PSO algorithm
6. Lặp lại cho đến khi tìm được best params
7. Lưu PSO results và best params

### Giai Đoạn 2: Full Training
1. Tạo agent với best params từ PSO
2. Train agent trên full training set
3. Lưu checkpoint theo interval
4. Evaluate trên test set mỗi 10 episodes
5. Track best models (train & test)
6. Lưu final model

---

## 💡 Tips & Best Practices

### PSO Configuration
- **Particles**: 5-10 là đủ cho most cases
- **Iterations**: 10-15 cho quick optimization, 20+ cho best results
- Tăng particles/iterations = chậm hơn nhưng tìm được params tốt hơn

### Training Configuration
- **Episodes**: 
  - 100-200 cho quick training
  - 500+ cho production models
- **Save Interval**: 10-20 episodes để không mất nhiều progress nếu crash

### Resume Training
- Luôn resume từ `pso_optimization_results.json` để giữ best params
- Có thể tăng số episodes khi resume để train lâu hơn

### Hardware
- **CPU**: ~2-3 giờ cho PSO + 100 episodes
- **GPU (CUDA)**: ~30-45 phút cho PSO + 100 episodes
- Khuyến nghị: Dùng GPU nếu có

---

## 📈 Monitoring Training

### Real-time Monitoring
- Console output mỗi episode
- Metrics: Reward, Average Reward, Loss, Epsilon
- Test evaluation mỗi 10 episodes

### Post-training Analysis
- Xem `training_progress.png` để thấy learning curve
- So sánh best train vs best test để detect overfitting
- Kiểm tra epsilon decay (should decrease gradually)

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Giảm batch size
python train_dqn_pso_lstm.py --particles 3 --iterations 8

# Hoặc dùng CPU
python train_dqn_pso_lstm.py --device cpu
```

### Training Too Slow
```bash
# Giảm PSO complexity
python train_dqn_pso_lstm.py --particles 3 --iterations 5

# Hoặc giảm episodes
python train_dqn_pso_lstm.py --episodes 50
```

### Poor Performance
```bash
# Tăng PSO optimization
python train_dqn_pso_lstm.py --particles 10 --iterations 20

# Tăng training episodes
python train_dqn_pso_lstm.py --episodes 500

# Thử coin khác
python train_dqn_pso_lstm.py --coin ETH
```

---

## 📚 Next Steps

Sau khi training xong:

1. **So sánh models**: Chạy `compare_all_models.py` để so sánh với LSTM, PSO+LSTM, PPO+PSO+LSTM
2. **Validate**: Chạy `validate_model.py` để test model trên unseen data
3. **Production**: Load best checkpoint và deploy

---

## 🎓 Ví Dụ Sử Dụng

### Quick Test (Fast)
```bash
python train_dqn_pso_lstm.py --particles 3 --iterations 5 --episodes 50
```

### Standard Training (Recommended)
```bash
python train_dqn_pso_lstm.py --particles 5 --iterations 10 --episodes 200
```

### Full Production Training
```bash
python train_dqn_pso_lstm.py --particles 10 --iterations 20 --episodes 500 --device cuda
```

### Multi-Coin Training
```bash
# BTC
python train_dqn_pso_lstm.py --coin BTC --checkpoint-dir checkpoints/dqn_pso_btc

# ETH
python train_dqn_pso_lstm.py --coin ETH --checkpoint-dir checkpoints/dqn_pso_eth

# BNB
python train_dqn_pso_lstm.py --coin BNB --checkpoint-dir checkpoints/dqn_pso_bnb
```

---

## ⚙️ Technical Details

### Model Architecture
- **Input**: Sequence of market features (OHLCV + indicators)
- **LSTM**: 1-3 layers với hidden size 64-256
- **Output**: Q-values cho 3 actions (Buy/Sell/Hold)
- **Optimizer**: Adam
- **Loss**: MSE

### PSO Hyperparameter Search Space
- `hidden_size`: 64 - 256
- `num_layers`: 1 - 3
- `sequence_length`: 12 - 48 (hours)
- `learning_rate`: 0.0001 - 0.005
- `gamma`: 0.95 - 0.999
- `epsilon_decay`: 0.990 - 0.999
- `batch_size`: 32 - 128
- `target_update_frequency`: 50 - 200

---

**Author**: AI Trading System  
**Version**: 1.0  
**Last Updated**: December 14, 2025
