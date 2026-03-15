// Bitcoin Trading Predictor - Main JavaScript

let priceChart = null;
let indicatorChart = null;
let updateInterval = null;
const UPDATE_INTERVAL = 10000; // 10 seconds

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Bitcoin Trading Predictor initialized');
    initializeCharts();
    fetchData(true); // Show loading only on first load
    // Attach coin selector change
    const coinSelect = document.getElementById('coinSelect');
    if (coinSelect) {
        coinSelect.addEventListener('change', function() {
            fetchData(true); // Show loading when user changes coin
        });
    }
    startAutoUpdate();
});

// Start automatic updates
function startAutoUpdate() {
    updateInterval = setInterval(fetchData, UPDATE_INTERVAL);
}

// Stop automatic updates
function stopAutoUpdate() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
}

// Fetch data from API
async function fetchData(showLoadingOverlay = false) {
    try {
        if (showLoadingOverlay) {
            showLoading();
        }
        // Determine selected symbol
        const coinSelect = document.getElementById('coinSelect');
        const symbol = coinSelect ? coinSelect.value : 'BTCUSDT';

        // Fetch prediction
        const predictionResponse = await fetch(`/api/predict?symbol=${symbol}&interval=1h`);
        const predictionData = await predictionResponse.json();
        
        if (!predictionData.success) {
            console.error('Error fetching prediction:', predictionData.error);
            hideLoading();
            return;
        }
        
        // Fetch history
    const historyResponse = await fetch(`/api/history?symbol=${symbol}&interval=1h&limit=100`);
        const historyData = await historyResponse.json();
        
        if (!historyData.success) {
            console.error('Error fetching history:', historyData.error);
            hideLoading();
            return;
        }
        
        // Update UI
        updatePriceInfo(predictionData.price);
        updatePrediction(predictionData.prediction);
        updateIndicators(predictionData.indicators);
        // updateRecommendation(predictionData.recommendation); // Removed recommendation card
        updateCharts(historyData.data);
        updateLastUpdateTime();
        
        if (showLoadingOverlay) {
            hideLoading();
        }
        
    } catch (error) {
        console.error('Error fetching data:', error);
        if (showLoadingOverlay) {
            hideLoading();
        }
    }
}

// Update price information
function updatePriceInfo(price) {
    document.getElementById('currentPrice').textContent = formatPrice(price.current);
    document.getElementById('openPrice').textContent = formatPrice(price.open);
    document.getElementById('highPrice').textContent = formatPrice(price.high);
    document.getElementById('lowPrice').textContent = formatPrice(price.low);
    document.getElementById('volume').textContent = formatVolume(price.volume);
    
    // Price change
    const priceChangeEl = document.getElementById('priceChange');
    const changePct = price.change_pct.toFixed(2);
    const changeClass = price.change >= 0 ? 'bg-success' : 'bg-danger';
    const changeIcon = price.change >= 0 ? '▲' : '▼';
    
    priceChangeEl.innerHTML = `
        <span class="badge ${changeClass}">
            ${changeIcon} ${Math.abs(changePct)}%
        </span>
    `;
    
    // Add pulse animation
    document.getElementById('currentPrice').classList.remove('price-up', 'price-down');
    document.getElementById('currentPrice').classList.add(price.change >= 0 ? 'price-up' : 'price-down');
}

// Update prediction
function updatePrediction(prediction) {
    const actionEl = document.getElementById('predictionAction');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    
    // Action badge
    let badgeClass = 'badge-hold';
    let actionIcon = '⏸️';
    
    if (prediction.action_name === 'MUA') {
        badgeClass = 'badge-buy';
        actionIcon = '📈';
    } else if (prediction.action_name === 'BÁN') {
        badgeClass = 'badge-sell';
        actionIcon = '📉';
    }
    
    actionEl.innerHTML = `<span class="badge ${badgeClass}">${actionIcon} ${prediction.action_name}</span>`;
    
    // Confidence bar
    const confidence = prediction.confidence.toFixed(1);
    confidenceBar.style.width = `${confidence}%`;
    confidenceText.textContent = `${confidence}%`;
    
    // Set bar color based on confidence
    confidenceBar.className = 'progress-bar';
    if (confidence >= 70) {
        confidenceBar.classList.add('bg-success');
    } else if (confidence >= 50) {
        confidenceBar.classList.add('bg-warning');
    } else {
        confidenceBar.classList.add('bg-danger');
    }
    
    // Update probabilities
    updateProbability('Buy', prediction.probabilities.buy);
    updateProbability('Sell', prediction.probabilities.sell);
    updateProbability('Hold', prediction.probabilities.hold);
}

// Update probability bars
function updateProbability(action, value) {
    const probEl = document.getElementById(`prob${action}`);
    const probBar = document.getElementById(`prob${action}Bar`);
    
    probEl.textContent = `${value.toFixed(1)}%`;
    probBar.style.width = `${value}%`;
}

// Update technical indicators
function updateIndicators(indicators) {
    // RSI
    const rsi = indicators.rsi.toFixed(1);
    document.getElementById('rsiValue').textContent = rsi;
    
    const rsiBar = document.getElementById('rsiBar');
    rsiBar.style.width = `${rsi}%`;
    
    // Set RSI color
    rsiBar.className = 'progress-bar';
    if (rsi < 30) {
        rsiBar.classList.add('rsi-oversold');
    } else if (rsi > 70) {
        rsiBar.classList.add('rsi-overbought');
    } else {
        rsiBar.classList.add('rsi-neutral');
    }
    
    // MACD
    document.getElementById('macdValue').textContent = indicators.macd.toFixed(2);
    document.getElementById('signalValue').textContent = indicators.signal.toFixed(2);
    
    // Bollinger Bands
    document.getElementById('bbUpper').textContent = formatPrice(indicators.bb_upper);
    document.getElementById('bbMiddle').textContent = formatPrice(indicators.bb_middle);
    document.getElementById('bbLower').textContent = formatPrice(indicators.bb_lower);
}

// Update recommendation
function updateRecommendation(rec) {
    const recHeader = document.getElementById('recHeader');
    const recAlert = document.getElementById('recAlert');
    const recTitle = document.getElementById('recTitle');
    const recMessage = document.getElementById('recMessage');
    const recAdvice = document.getElementById('recAdvice');
    const recDetails = document.getElementById('recDetails');
    const riskBadge = document.getElementById('riskBadge');
    
    // Set alert color based on action only
    let alertClass = 'alert-warning';
    
    if (rec.title.includes('MUA')) {
        alertClass = 'alert-success';
    } else if (rec.title.includes('BÁN')) {
        alertClass = 'alert-danger';
    }
    
    // Keep header with single color - no change
    recAlert.className = `alert ${alertClass}`;
    
    // Set content
    recTitle.innerHTML = `<i class="fas fa-lightbulb"></i> ${rec.title}`;
    recMessage.textContent = rec.message;
    recAdvice.textContent = rec.advice;
    
    // Risk badge
    let riskClass = 'risk-medium';
    if (rec.risk_level === 'THẤP') {
        riskClass = 'risk-low';
    } else if (rec.risk_level === 'CAO') {
        riskClass = 'risk-high';
    }
    
    riskBadge.className = `badge ${riskClass}`;
    riskBadge.textContent = rec.risk_level;
    
    // Details
    recDetails.innerHTML = '';
    rec.details.forEach(detail => {
        const li = document.createElement('li');
        li.textContent = detail;
        recDetails.appendChild(li);
    });
}

// Initialize charts
function initializeCharts() {
    // Price Chart
    const priceCtx = document.getElementById('priceChart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Giá (USD)',
                    data: [],
                    borderColor: '#2962FF',
                    backgroundColor: 'rgba(41, 98, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'BB Upper',
                    data: [],
                    borderColor: '#0ECB81',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false
                },
                {
                    label: 'BB Lower',
                    data: [],
                    borderColor: '#F6465D',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
    
    // Indicator Chart
    const indicatorCtx = document.getElementById('indicatorChart').getContext('2d');
    indicatorChart = new Chart(indicatorCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'RSI',
                    data: [],
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    borderWidth: 2,
                    yAxisID: 'y',
                    tension: 0.4
                },
                {
                    label: 'MACD',
                    data: [],
                    borderColor: '#17a2b8',
                    backgroundColor: 'rgba(23, 162, 184, 0.1)',
                    borderWidth: 2,
                    yAxisID: 'y1',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'RSI'
                    },
                    min: 0,
                    max: 100
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'MACD'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Update charts with new data
function updateCharts(data) {
    if (!data || data.length === 0) return;
    
    // Take last 50 data points for better visualization
    const displayData = data.slice(-50);
    
    // Update price chart
    const labels = displayData.map(d => formatDateTime(d.timestamp));
    const prices = displayData.map(d => d.close);
    const bbUpper = displayData.map(d => d.bb_upper);
    const bbLower = displayData.map(d => d.bb_lower);
    
    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = prices;
    priceChart.data.datasets[1].data = bbUpper;
    priceChart.data.datasets[2].data = bbLower;
    priceChart.update('none');
    
    // Update indicator chart
    const rsiData = displayData.map(d => d.rsi);
    const macdData = displayData.map(d => d.macd);
    
    indicatorChart.data.labels = labels;
    indicatorChart.data.datasets[0].data = rsiData;
    indicatorChart.data.datasets[1].data = macdData;
    indicatorChart.update('none');
}

// Update last update time
function updateLastUpdateTime() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('vi-VN');
    document.getElementById('lastUpdate').textContent = `Cập nhật lần cuối: ${timeStr}`;
}

// Format price
function formatPrice(price) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(price);
}

// Format volume
function formatVolume(volume) {
    if (volume >= 1000000) {
        return (volume / 1000000).toFixed(2) + 'M';
    } else if (volume >= 1000) {
        return (volume / 1000).toFixed(2) + 'K';
    }
    return volume.toFixed(2);
}

// Format date time
function formatDateTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('vi-VN', { 
        hour: '2-digit', 
        minute: '2-digit'
    });
}

// Show loading overlay
function showLoading() {
    document.getElementById('loadingOverlay').classList.remove('hidden');
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

// Handle page visibility change
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        stopAutoUpdate();
    } else {
        fetchData();
        startAutoUpdate();
    }
});
