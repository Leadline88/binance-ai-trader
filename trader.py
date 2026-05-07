import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import torch

from config import Config
from binance_client import BinanceClient
from indicators import TechnicalIndicators
from model import TradingLSTM, ModelTrainer
from backtester import Backtester, Validator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trader.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIBinanceTrader:
    def __init__(self, config: Config):
        self.config = config
        self.client = BinanceClient(config)
        self.backtester = Backtester(
            risk_per_trade=config.RISK_PER_TRADE
        )
        self.validator = Validator(min_sharpe=config.MIN_SHARPE)
        
        # Feature columns for model
        self.feature_cols = [
            'returns', 'log_returns', 'rsi_norm', 'bb_position', 
            'macd_hist', 'volume_change', 'sma_20', 'ema_12'
        ]
        
        # Init model
        input_dim = len(self.feature_cols)
        self.model = TradingLSTM(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM
        )
        self.trainer = ModelTrainer(self.model, lr=config.LEARNING_RATE)
        
        # Load existing model if available
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        self.trainer.load(config.MODEL_PATH)
        
        self.last_retrain = datetime.now() - timedelta(hours=config.RETRAIN_EVERY_HOURS + 1)
        self.position = 0.0
        self.equity_peak = 10000.0  # will be updated from balance
        self.daily_pnl = 0.0
        self.last_day = datetime.now().date()
        
        logger.info("AI Binance Trader initialized successfully")
    
    def fetch_and_prepare_data(self, limit: int = 500) -> pd.DataFrame:
        """Fetch latest candles and add indicators"""
        try:
            raw_data = self.client.get_klines(
                self.config.SYMBOL, 
                self.config.INTERVAL, 
                limit=limit,
                futures=True
            )
            df = pd.DataFrame(raw_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df = TechnicalIndicators.add_all_indicators(df)
            
            # Ensure all feature cols exist
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            raise
    
    def should_retrain(self) -> bool:
        hours_since = (datetime.now() - self.last_retrain).total_seconds() / 3600
        return hours_since >= self.config.RETRAIN_EVERY_HOURS
    
    def retrain_model(self, df: pd.DataFrame):
        """Periodic full retrain + validation"""
        logger.info("Starting model retraining...")
        
        try:
            # Train
            history = self.trainer.train(
                df, 
                seq_len=self.config.SEQ_LEN,
                feature_cols=self.feature_cols,
                epochs=15,
                batch_size=32
            )
            
            # Backtest to validate
            bt_results = self.backtester.run_backtest(
                df, self.model, self.config.SEQ_LEN, self.feature_cols
            )
            
            # Validate (this is where the old -4 bug was fixed!)
            self.validator.validate(bt_results)
            
            # Save if good
            self.trainer.save(self.config.MODEL_PATH)
            self.last_retrain = datetime.now()
            
            logger.info(f"Retraining successful! Sharpe: {bt_results['sharpe_ratio']:.2f}")
            
        except ValueError as ve:
            logger.warning(f"Validation failed after retrain: {ve}. Keeping old model.")
        except Exception as e:
            logger.error(f"Retrain error: {e}")
    
    def fine_tune_on_latest(self, df: pd.DataFrame):
        """Light update with newest data"""
        try:
            recent_df = df.tail(100)  # last ~4 days on 1h
            self.trainer.fine_tune(recent_df, self.config.SEQ_LEN, self.feature_cols, epochs=3)
            logger.info("Fine-tuning completed")
        except Exception as e:
            logger.warning(f"Fine-tune skipped: {e}")
    
    def get_signal(self, df: pd.DataFrame) -> float:
        """Get predicted return from latest sequence"""
        if len(df) < self.config.SEQ_LEN + 5:
            return 0.0
        
        features = df[self.feature_cols].iloc[-self.config.SEQ_LEN:].values.astype(np.float32)
        
        # Normalize
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features_norm = (features - mean) / std
        
        x = torch.tensor(features_norm, dtype=torch.float32)
        pred = self.trainer.model.predict(x)
        return pred
    
    def risk_check(self) -> bool:
        """Circuit breaker for daily loss"""
        today = datetime.now().date()
        if today != self.last_day:
            self.daily_pnl = 0.0
            self.last_day = today
        
        current_balance = self.client.get_account_balance(futures=True)
        if self.equity_peak == 0:
            self.equity_peak = current_balance
        
        daily_loss_pct = (self.daily_pnl / self.equity_peak) * 100 if self.equity_peak > 0 else 0
        
        if daily_loss_pct < -self.config.MAX_DAILY_LOSS * 100:
            logger.critical(f"DAILY LOSS LIMIT HIT: {daily_loss_pct:.2f}% ! Stopping trading.")
            # Close position
            self.client.close_position(self.config.SYMBOL)
            return False
        return True
    
    def execute_trade(self, signal: float, current_price: float):
        """Execute based on signal with proper risk"""
        if not self.risk_check():
            return
        
        pos_info = self.client.get_position(self.config.SYMBOL)
        current_pos = pos_info['positionAmt']
        
        # Determine action
        action = None
        if signal > self.config.BUY_THRESHOLD and current_pos <= 0.01:
            action = 'BUY'
        elif signal < self.config.SELL_THRESHOLD and current_pos >= -0.01:
            action = 'SELL'
        
        if not action:
            return
        
        balance = self.client.get_account_balance()
        if balance < 10:
            logger.warning("Insufficient balance")
            return
        
        # Position size: risk based (use ATR for stop distance if possible)
        risk_amount = balance * self.config.RISK_PER_TRADE
        # Simple: use 1% of balance for now (can improve with ATR)
        quantity = (risk_amount / current_price) * self.config.LEVERAGE
        
        if quantity < 0.001:  # min order
            return
        
        try:
            order = self.client.place_order(
                self.config.SYMBOL, 
                action, 
                'MARKET', 
                quantity=abs(quantity),
                futures=True
            )
            if order:
                logger.info(f"Executed {action} {quantity:.4f} {self.config.SYMBOL} @ ~{current_price}")
                self.daily_pnl += -risk_amount if action == 'SELL' else 0  # rough tracking
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
    
    def run(self):
        """Main trading loop - continuous learning + trading"""
        logger.info(f"Starting live trading for {self.config.SYMBOL} on {self.config.INTERVAL}...")
        
        while True:
            try:
                # 1. Fetch data
                df = self.fetch_and_prepare_data(limit=300)
                current_price = df['close'].iloc[-1]
                
                # 2. Check if retrain needed (daily or config)
                if self.should_retrain():
                    self.retrain_model(df)
                else:
                    # Light update
                    self.fine_tune_on_latest(df)
                
                # 3. Get AI signal
                signal = self.get_signal(df)
                logger.info(f"Current price: {current_price:.2f} | AI predicted return: {signal*100:.3f}%")
                
                # 4. Execute if strong signal
                self.execute_trade(signal, current_price)
                
                # 5. Sleep based on interval
                sleep_seconds = {
                    '1m': 60, '5m': 300, '15m': 900, 
                    '1h': 3600, '4h': 14400, '1d': 86400
                }.get(self.config.INTERVAL, 3600)
                
                # Sleep with small jitter
                time.sleep(sleep_seconds + np.random.randint(-30, 30))
                
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                self.client.close_position(self.config.SYMBOL)
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(60)  # backoff


if __name__ == "__main__":
    config = Config()
    trader = AIBinanceTrader(config)
    trader.run()
