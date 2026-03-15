# 🌐 Web Application - DQN Trading Predictor

Real-time cryptocurrency trading predictions using Deep Q-Network (DQN).

## 🚀 Quick Start

```bash
cd web
python app.py
```

Visit: **http://localhost:5000**

## ✨ Features

- 📊 **Real-time Data**: Fetches live prices from Binance API
- 🤖 **DQN Predictions**: AI-powered BUY/SELL/HOLD recommendations
- 📈 **Technical Indicators**: RSI, MACD, Bollinger Bands, ADX
- 💯 **Confidence Score**: Shows prediction confidence (%)
- 🎨 **Interactive UI**: Responsive Bootstrap design

## 📦 Requirements

```bash
# From project root
pip install -r requirements.txt
```

Dependencies:
- Flask
- Flask-CORS
- PyTorch (for DQN)
- NumPy, Pandas
- Binance API (via python-binance or requests)

## 🔧 Configuration

### Update DQN Model

The web app automatically loads the latest DQN checkpoint:
```python
MODEL_PATH = 'src/checkpoints_dqn/checkpoint_latest.pkl'
```

To use a specific checkpoint:
```python
# Edit web/app.py line 36
MODEL_PATH = 'src/checkpoints_dqn/checkpoint_episode_5000.pkl'
```

### Change Symbol

Default: Bitcoin (BTCUSDT)

To predict other cryptocurrencies, edit in `web/app.py`:
```python
def get_realtime_data(symbol='ETHUSDT', interval='1h', limit=100):
```

Supported symbols:
- `BTCUSDT` - Bitcoin
- `ETHUSDT` - Ethereum
- `BNBUSDT` - Binance Coin
- `SOLUSDT` - Solana
- `ADAUSDT` - Cardano

## 📊 API Endpoints

### GET `/api/predict`
Get trading prediction

**Response:**
```json
{
  "success": true,
  "prediction": {
    "action": 0,
    "action_name": "MUA",
    "confidence": 85.3,
    "probabilities": {
      "buy": 85.3,
      "sell": 8.2,
      "hold": 6.5
    },
    "q_values": {
      "buy": 2.45,
      "sell": -0.82,
      "hold": -1.23
    }
  },
  "current_price": 65432.50,
  "indicators": {
    "rsi": 58.3,
    "macd": 234.5,
    "adx": 28.7
  }
}
```

### GET `/api/status`
Check system status

**Response:**
```json
{
  "success": true,
  "model_loaded": true,
  "fetcher_ready": true,
  "timestamp": "2025-11-05T15:30:00"
}
```

## 🎯 Understanding Predictions

### Actions
- **MUA (BUY)**: Model predicts price will increase
- **BÁN (SELL)**: Model predicts price will decrease  
- **GIỮ (HOLD)**: Model suggests waiting

### Confidence
- **>70%**: High confidence - Strong signal
- **50-70%**: Moderate confidence - Consider other factors
- **<50%**: Low confidence - Use caution

### Q-Values
- Represents expected future reward for each action
- Higher Q-value = Better expected outcome
- Action with highest Q-value is chosen

## 🔍 Troubleshooting

### Model not loaded
```bash
❌ Error: DQN model not found
```

**Solution:** Train the model first
```bash
cd src
python main_multi_coin_dqn.py --episodes 5000
```

### Binance connection error
```bash
❌ Error fetching data from Binance
```

**Solutions:**
1. Check internet connection
2. Binance API may be rate-limited (wait 1 minute)
3. Check firewall/proxy settings

### Port already in use
```bash
Address already in use: Port 5000
```

**Solution:** Change port in `app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use different port
```

## 📈 Performance

Based on Episode 276 validation:
- **Accuracy**: 80% coins profitable (4/5)
- **Avg Ann.Return**: 72.24%
- **Avg MDD**: 32.22%
- **Action Balance**: HOLD 48%, BUY 45%, SELL 7%

## 🛡️ Disclaimer

**⚠️ FOR EDUCATIONAL PURPOSES ONLY**

This is an AI trading assistant for learning and research. Do NOT use for real trading without:
- Thorough backtesting
- Risk management strategy
- Professional financial advice
- Understanding of cryptocurrency risks

Past performance does not guarantee future results.

## 📝 Development

### Run in debug mode
```bash
python app.py
```

### Run in production
```bash
# Using gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or using waitress (Windows)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Enable CORS for external access
Already enabled via Flask-CORS. To restrict:
```python
# In app.py
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
```

## 🎨 Customization

### Change UI Theme
Edit `static/css/style.css`

### Add More Indicators
1. Calculate in `/api/predict` endpoint
2. Add to response JSON
3. Display in `templates/index.html`

### Multi-Symbol Support
Modify UI to allow symbol selection:
```html
<select id="symbol">
  <option value="BTCUSDT">Bitcoin</option>
  <option value="ETHUSDT">Ethereum</option>
</select>
```

## 📚 Related Files

- `app.py` - Main Flask application
- `templates/index.html` - Web UI
- `static/css/style.css` - Styling
- `static/js/main.js` - Frontend logic
- `../src/models/dqn_agent.py` - DQN model
- `../validate_realtime.py` - Real-time validation

## 🤝 Contributing

To improve the web app:
1. Add more cryptocurrencies
2. Implement user accounts
3. Add historical charts
4. Create mobile app version
5. Add email/SMS alerts

---

**Made with ❤️ using DQN + Flask**
