## 5️⃣ Câu 5: Tại Sao Dùng Q-Learning/DQN (KHÔNG PPO)?

### ⚠️ Điều Chỉnh: Dự Án KHÔNG Dùng PPO

**Nhưng tại sao lại chọn Q-Learning thay vì PPO?**

### So Sánh Chi Tiết: Q-Learning vs Các Thuật Toán Khác

#### **1. Q-Learning vs PPO**

| Tiêu Chí | Q-Learning (Dự Án ✅) | PPO |
|---------|-------|-----|
| **Action Space** | Discrete ✅ (3 actions) | Continuous (không phù hợp) |
| **Data Efficiency** | Off-policy → Reuse old data | On-policy → Phải generate new data |
| **Sample Efficiency** | Cao (Experience Replay) | Thấp hơn (sampling overhead) |
| **Convergence Speed** | Nhanh (Bellman target) | Chậm (multiple epochs) |
| **Code Complexity** | Thấp | Cao (Actor + Critic + PPO clipping) |
| **GPU Memory** | Thấp (64 batch) | Cao (large batch cần stability) |
| **Interpretability** | Cao (Q-values = hành động value) | Thấp (policy output = probability) |

**Kết Luận**: **Q-Learning phù hợp hơn** cho discrete action space trading

#### **2. Q-Learning vs Policy Gradient (PG)**

```
Policy Gradient (PG):
- Tối ưu policy trực tiếp: θ ← θ + α∇log(π(a|s))R
- Mục tiêu: Maximize E[Reward]
- Vấn đề: High variance, slow convergence

Q-Learning:
- Tối ưu Q-values: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- Mục tiêu: Learn value function (stable baseline)
- Ưu điểm: Low variance, fast convergence
```

#### **3. Q-Learning vs Actor-Critic**

| Khía Cạnh | Q-Learning (Dự Án) | Actor-Critic |
|-----------|--------|---------|
| **Components** | Q-Network = 1 network | Actor network + Critic network = 2 networks |
| **Complexity** | Đơn giản | Phức tạp |
| **Training Stability** | Cao | Cần fine-tune 2 networks |
| **Gradient Variance** | Low (Bellman target) | Medium (policy + value gradient) |
| **For Trading** | ✅ Đủ tốt | ⚠️ Overkill |

#### **4. Q-Learning vs SARSA**

```
Q-Learning (Off-Policy):
Q(s,a) ← Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
- Học từ optimal policy (max Q)
- Lebih aggressive exploration

SARSA (On-Policy):
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a_next) - Q(s,a)]
- Học từ actual action taken
- Lebih conservative

Dự án chọn Q-Learning vì aggressive learning cần cho volatile market
```

### Tại Sao Q-Learning Là Lựa Chọn Tốt Nhất?

#### **Lý Do 1: Action Space Discrete**
```python
# Trading chỉ có 3 hành động rõ ràng
action_space = {0: Buy, 1: Sell, 2: Hold}
# Q-Learning optimal cho discrete space
# PPO được design cho continuous actions
```

#### **Lý Do 2: Sample Efficiency (Off-Policy)**
```python
# Q-Learning: Off-Policy
# Có thể học từ experience buffer cũ
# Không phải generate new data sau mỗi epoch

# PPO: On-Policy
# Phải generate new trajectories mỗi epoch
# Inefficient cho single-market training
```

#### **Lý Do 3: Stability Cho Trading**
```python
# Experience Replay + Target Network
# = Cực kỳ stable cho financial data
# PPO cần PPO clipping (thêm hyperparameter)

reward_network = dqn_network  # Learn Q-values
target_network = dqn_network.copy()  # Stable target
# Update target mỗi 1000 steps
```

#### **Lý Do 4: Interpretability**
```python
# Q-Learning: Output = Q-values
Q[Buy] = 0.5 (Medium good)
Q[Sell] = -0.1 (Bad)
Q[Hold] = 0.2 (OK)
# Action = argmax(Q) = Buy (đơn giản)

# PPO: Output = Policy probability
π[Buy] = 0.6
π[Sell] = 0.1
π[Hold] = 0.3
# Action = sample from π (phức tạp hơn)
```

#### **Lý Do 5: Real-Time Performance**
```python
# DQN inference: 1 forward pass
next_action = agent.select_action(state)

# PPO inference: Có thể cần multiple forward passes
# cho stability (tìm best action)
# Trading thực tế cần decision nhanh < 100ms
```

### Kết Luận

**Q-Learning/DQN là lựa chọn tối ưu vì:**
1. ✅ Phù hợp discrete action space
2. ✅ High sample efficiency (off-policy)
3. ✅ Stability cao (Target Network + Replay)
4. ✅ Fast convergence (Bellman equation)
5. ✅ Interpretability (Q-values)
6. ✅ Real-time inference (1 forward pass)
7. ✅ Proven cho RL trading (DQN state-of-art 2015+)

