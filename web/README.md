# Bitcoin Trading Predictor - Web Application

## 🌟 Giới Thiệu

Ứng dụng web dự đoán giao dịch Bitcoin theo thời gian thực sử dụng AI (Q-Learning) đã được train từ dữ liệu lịch sử.

### Tính năng chính:

1. **Dự đoán theo thời gian thực**: Lấy dữ liệu mới nhất từ Binance API và dự đoán hành động (Mua/Bán/Giữ)
2. **Hiển thị độ tin cậy**: Tỷ lệ phần trăm độ tin cậy của dự đoán
3. **Phân tích kỹ thuật**: Hiển thị các chỉ báo RSI, MACD, Bollinger Bands
4. **Biểu đồ trực quan**: Biểu đồ giá và chỉ báo kỹ thuật real-time
5. **Khuyến nghị chi tiết**: Lời khuyên cụ thể về việc nên mua/bán/giữ
6. **Cập nhật tự động**: Tự động refresh dữ liệu mỗi 10 giây

## 📋 Yêu Cầu

### 1. Model đã được train
Đảm bảo bạn đã chạy training và có file model tại:
```
data/processed/q_learning_model_combined.pkl
```

Nếu chưa có, hãy chạy:
```bash
cd src
python main_gpu.py
```

(Sau khi train xong, có thể chạy `python main_enhanced.py` để so sánh với data thực tế)

### 2. Python packages
Cài đặt các thư viện cần thiết:
```bash
cd web
pip install -r requirements.txt
```

hoặc sử dụng requirements.txt từ thư mục gốc:
```bash
pip install flask flask-cors
```

### 3. Binance API (Tùy chọn)
- Nếu để trống API key trong `src/config/config.py`, ứng dụng sẽ dùng public API của Binance
- Hoặc điền API key để có rate limit cao hơn

## 🚀 Cách Chạy

### Bước 1: Activate môi trường Python
```bash
conda activate MeoMeo
```

### Bước 2: Chuyển vào thư mục web
```bash
cd web
```

### Bước 3: Chạy ứng dụng Flask
```bash
python app.py
```

### Bước 4: Mở trình duyệt
Truy cập: **http://localhost:5000**

## 📱 Giao Diện

### Màn hình chính gồm:

#### Cột Trái:
- **Giá Hiện Tại**: Hiển thị giá Bitcoin real-time và % thay đổi
- **Dự Đoán AI**: Khuyến nghị MUA/BÁN/GIỮ với độ tin cậy
- **Chỉ Báo Kỹ Thuật**: RSI, MACD, Bollinger Bands

#### Cột Phải:
- **Khuyến Nghị**: Lời khuyên chi tiết và phân tích
- **Biểu Đồ Giá**: Đường giá với Bollinger Bands
- **Biểu Đồ Chỉ Báo**: RSI và MACD theo thời gian

## 🎯 API Endpoints

### 1. GET `/api/predict`
Dự đoán hành động giao dịch

**Parameters:**
- `symbol` (optional): Mã coin, mặc định `BTCUSDT`
- `interval` (optional): Khung thời gian, mặc định `1h`

**Response:**
```json
{
    "success": true,
    "timestamp": "2025-10-29T...",
    "price": {
        "current": 67890.50,
        "change_pct": 2.5
    },
    "prediction": {
        "action_name": "MUA",
        "confidence": 75.5,
        "probabilities": {
            "buy": 75.5,
            "sell": 12.3,
            "hold": 12.2
        }
    },
    "recommendation": {
        "title": "📈 KHUYẾN NGHỊ MUA",
        "message": "Model dự đoán xu hướng tăng...",
        "advice": "Đây có thể là thời điểm tốt..."
    }
}
```

### 2. GET `/api/history`
Lấy dữ liệu lịch sử giá

**Parameters:**
- `symbol` (optional): Mã coin
- `interval` (optional): Khung thời gian
- `limit` (optional): Số lượng data points, mặc định 100

**Response:**
```json
{
    "success": true,
    "data": [
        {
            "timestamp": "2025-10-29T...",
            "open": 67000,
            "high": 68000,
            "low": 66500,
            "close": 67500,
            "volume": 1234567,
            "rsi": 55.5,
            "macd": 123.45
        }
    ]
}
```

### 3. GET `/api/status`
Kiểm tra trạng thái hệ thống

**Response:**
```json
{
    "success": true,
    "model_loaded": true,
    "fetcher_ready": true,
    "timestamp": "2025-10-29T..."
}
```

## 🔧 Tùy Chỉnh

### Thay đổi tần suất cập nhật
Trong file `static/js/main.js`, sửa dòng:
```javascript
const UPDATE_INTERVAL = 10000; // 10 giây
```

### Thay đổi số lượng data points hiển thị
Trong file `static/js/main.js`, sửa hàm `updateCharts`:
```javascript
const displayData = data.slice(-50); // 50 điểm cuối
```

### Thay đổi coin symbol
Trong file `static/js/main.js`, sửa các lời gọi API:
```javascript
await fetch('/api/predict?symbol=ETHUSDT&interval=1h');
```

## ⚠️ Lưu Ý Quan Trọng

1. **Đây KHÔNG phải lời khuyên tài chính**: Ứng dụng chỉ mang tính chất học tập và nghiên cứu
2. **Model có thể sai**: Dự đoán AI không đảm bảo 100% chính xác
3. **Luôn DYOR**: Tự nghiên cứu trước khi đưa ra quyết định đầu tư
4. **Quản lý rủi ro**: Chỉ đầu tư số tiền bạn có thể chấp nhận mất
5. **Rate Limit**: API Binance có giới hạn số lượng request, nếu gặp lỗi 429, hãy tăng `UPDATE_INTERVAL`

## 🐛 Xử Lý Lỗi

### Lỗi: "Model file not found"
**Giải pháp**: Chạy training trước:
```bash
cd src
python main_enhanced.py
```

### Lỗi: "Cannot connect to Binance"
**Giải pháp**: 
- Kiểm tra kết nối internet
- Kiểm tra API key trong `src/config/config.py`
- Hoặc để trống API key để dùng public API

### Lỗi: "Module not found"
**Giải pháp**: 
```bash
pip install flask flask-cors
```

### Lỗi 429: "Rate limit exceeded"
**Giải pháp**: 
- Tăng `UPDATE_INTERVAL` trong `main.js`
- Hoặc thêm API key vào config

## 📈 Nâng Cấp Tương Lai

- [ ] Thêm hỗ trợ nhiều coin khác nhau
- [ ] Lưu lịch sử dự đoán vào database
- [ ] Thêm chế độ paper trading để test chiến lược
- [ ] Tích hợp notification khi có tín hiệu mua/bán
- [ ] Thêm backtesting với dữ liệu lịch sử
- [ ] Mobile responsive design
- [ ] Dark mode
- [ ] Multi-language support

## 📞 Hỗ Trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra console log trong browser (F12)
2. Kiểm tra terminal log của Flask
3. Đảm bảo model đã được train
4. Đảm bảo tất cả dependencies đã được cài đặt

## 📄 License

MIT License - Sử dụng cho mục đích học tập và nghiên cứu.

---

**⚡ Quick Start:**
```bash
conda activate MeoMeo
cd web
python app.py
# Mở browser: http://localhost:5000
```
