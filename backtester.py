import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from model import TradingLSTM, ModelTrainer
from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.0004, 
                 slippage: float = 0.0001, risk_per_trade: float = 0.01):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate  # ~0.04% for futures maker/taker avg
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0, 
                               periods_per_year: int = 365) -> float:
        """Correct Sharpe ratio calculation for crypto (24/7 market)"""
        if len(returns) < 2:
            return 0.0
        
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]  # remove NaNs
        
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        mean_excess = np.mean(excess_returns)
        std_dev = np.std(excess_returns, ddof=1)  # sample std
        
        if std_dev < 1e-10:
            return 0.0
        
        sharpe = (mean_excess / std_dev) * np.sqrt(periods_per_year)
        return float(sharpe)
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0,
                                periods_per_year: int = 365) -> float:
        """Sortino: only downside deviation"""
        if len(returns) < 2:
            return 0.0
        returns = np.asarray(returns)
        downside = returns[returns < 0]
        if len(downside) < 2:
            return self.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        
        excess = returns - (risk_free_rate / periods_per_year)
        mean_excess = np.mean(excess)
        downside_std = np.std(downside, ddof=1)
        if downside_std < 1e-10:
            return 0.0
        return (mean_excess / downside_std) * np.sqrt(periods_per_year)
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Max drawdown in percent"""
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return float(np.min(drawdown) * 100)
    
    def calculate_calmar_ratio(self, returns: np.ndarray, equity: np.ndarray) -> float:
        """Calmar = CAGR / MaxDD"""
        if len(returns) < 2:
            return 0.0
        total_return = (equity[-1] / equity[0]) - 1
        years = len(returns) / 365.0
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        mdd = abs(self.calculate_max_drawdown(equity) / 100)
        return cagr / mdd if mdd > 0 else 0.0
    
    def run_backtest(self, df: pd.DataFrame, model: TradingLSTM, seq_len: int, 
                     feature_cols: list, initial_capital: float = None) -> Dict[str, Any]:
        """Run walk-forward style backtest with AI signals"""
        if initial_capital is None:
            initial_capital = self.initial_capital
            
        df = df.copy().reset_index(drop=True)
        capital = initial_capital
        position = 0.0  # in base asset (BTC etc)
        entry_price = 0.0
        equity_curve = [capital]
        trades = []
        daily_returns = []
        
        trainer = ModelTrainer(model)  # for prediction only
        
        for i in range(seq_len, len(df) - 1):
            current_price = df.loc[i, 'close']
            features = df.loc[i-seq_len:i-1, feature_cols].values.astype(np.float32)
            
            # Normalize like in training (simple, in real use better scaler)
            feat_mean = features.mean(axis=0)
            feat_std = features.std(axis=0) + 1e-8
            features_norm = (features - feat_mean) / feat_std
            
            x = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
            pred_return = trainer.model.predict(x[0]) if hasattr(trainer.model, 'predict') else 0.0
            
            # Signal logic with risk management
            signal = 0
            if pred_return > 0.005 and position == 0:  # strong buy
                signal = 1
            elif pred_return < -0.005 and position > 0:  # strong sell
                signal = -1
            
            # Execute
            if signal == 1 and capital > 10:
                # Risk based position size (simplified % of capital)
                risk_amount = capital * self.risk_per_trade
                position_size = risk_amount / current_price  # approx, ignore stop for backtest
                cost = position_size * current_price * (1 + self.fee_rate + self.slippage)
                if cost <= capital:
                    capital -= cost
                    position += position_size
                    entry_price = current_price
                    trades.append({'type': 'BUY', 'price': current_price, 'size': position_size, 'time': i})
            
            elif signal == -1 and position > 0:
                # Sell all
                sell_value = position * current_price * (1 - self.fee_rate - self.slippage)
                capital += sell_value
                trades.append({'type': 'SELL', 'price': current_price, 'size': position, 'time': i, 'pnl': (current_price - entry_price) * position})
                position = 0.0
            
            # Update equity (mark to market)
            current_equity = capital + position * current_price
            equity_curve.append(current_equity)
            
            # Daily return approx (for metrics)
            if len(equity_curve) > 1:
                daily_returns.append((equity_curve[-1] / equity_curve[-2]) - 1)
        
        # Final close if position open
        if position > 0:
            final_price = df.iloc[-1]['close']
            sell_value = position * final_price * (1 - self.fee_rate)
            capital = capital + sell_value
            equity_curve[-1] = capital
        
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd = self.calculate_max_drawdown(equity_curve)
        calmar = self.calculate_calmar_ratio(returns, equity_curve)
        
        win_rate = 0.0
        if trades:
            profitable = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = profitable / len([t for t in trades if 'pnl' in t]) if any('pnl' in t for t in trades) else 0.5
        
        results = {
            'final_capital': capital,
            'total_return_pct': ((capital / initial_capital) - 1) * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'equity_curve': equity_curve.tolist(),
            'trades': trades[-5:] if trades else []  # last 5 for summary
        }
        
        logger.info(f"Backtest completed: Sharpe={sharpe:.2f}, Return={results['total_return_pct']:.1f}%, MaxDD={max_dd:.1f}%")
        return results


class Validator:
    """Validates model performance - FIXED the -4 Sharpe bug here"""
    
    def __init__(self, min_sharpe: float = 0.8, max_dd: float = 25.0, min_trades: int = 20):
        self.min_sharpe = min_sharpe
        self.max_dd = max_dd
        self.min_trades = min_trades
    
    def validate(self, backtest_results: Dict[str, Any]) -> bool:
        sharpe = backtest_results.get('sharpe_ratio', 0.0)
        max_dd = abs(backtest_results.get('max_drawdown_pct', 0.0))
        num_trades = backtest_results.get('num_trades', 0)
        
        errors = []
        
        if sharpe < self.min_sharpe:
            errors.append(f"Sharpe ratio too low: {sharpe:.2f} < {self.min_sharpe}")
        if max_dd > self.max_dd:
            errors.append(f"Max Drawdown too high: {max_dd:.1f}% > {self.max_dd}%")
        if num_trades < self.min_trades:
            errors.append(f"Too few trades for reliable validation: {num_trades} < {self.min_trades}")
        
        if errors:
            error_msg = " | ".join(errors)
            logger.error(f"Validation FAILED: {error_msg}")
            # NO hardcoded -4 anymore! Proper error instead of magic number
            raise ValueError(f"Model validation failed: {error_msg}")
        
        logger.info(f"Validation PASSED: Sharpe={sharpe:.2f}, MaxDD={max_dd:.1f}%, Trades={num_trades}")
        return True
