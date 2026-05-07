import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    USE_TESTNET = os.getenv('USE_TESTNET', 'True').lower() == 'true'
    
    # Trading
    SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
    INTERVAL = os.getenv('INTERVAL', '1h')  # 1m, 5m, 15m, 1h, 4h, 1d
    LEVERAGE = int(os.getenv('LEVERAGE', '1'))  # for futures, 1-20
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))  # 1% of capital
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '0.03'))  # 3% circuit breaker
    
    # Model
    SEQ_LEN = int(os.getenv('SEQ_LEN', '24'))  # lookback candles
    HIDDEN_DIM = int(os.getenv('HIDDEN_DIM', '128'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
    RETRAIN_EVERY_HOURS = int(os.getenv('RETRAIN_EVERY_HOURS', '24'))
    
    # Thresholds
    BUY_THRESHOLD = float(os.getenv('BUY_THRESHOLD', '0.005'))  # predicted return > 0.5%
    SELL_THRESHOLD = float(os.getenv('SELL_THRESHOLD', '-0.005'))
    MIN_SHARPE = float(os.getenv('MIN_SHARPE', '0.8'))
    
    # Paths
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/ai_trader.pth')
    LOG_PATH = os.getenv('LOG_PATH', 'logs/trader.log')
    DATA_PATH = os.getenv('DATA_PATH', 'data/historical.csv')
