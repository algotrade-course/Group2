import optuna
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.indicators.bollinger_bands import add_bollinger_bands
from src.indicators.rsi import add_rsi
from src.indicators.atr import add_atr
from src.strategy.signal_generator import SignalGenerator
from src.backtest.backtest_engine import BacktestEngine
from src.strategy.risk_manager import RiskManager
from src.backtest.performance import PerformanceMetrics

class StrategyOptimizer:
    def __init__(self, train_data, test_data=None, initial_capital=100000, n_trials=100,timeframes=None):
        self.train_data = train_data
        self.test_data = test_data
        self.initial_capital = initial_capital
        self.n_trials = n_trials
        self.processor = DataProcessor()
        self.backtest_engine = BacktestEngine(initial_balance=initial_capital)
        self.risk_manager = RiskManager()
        self.logger = self._setup_logger()
        self.timeframes = timeframes if timeframes else ['5min', '15min', '30min', '1h', '4h', '1d']
        
    def _setup_logger(self):
        logger = logging.getLogger("strategy_optimizer")
        logger.setLevel(logging.INFO)
        
        os.makedirs("logs", exist_ok=True)
        
        if not logger.handlers:
            file_handler = logging.FileHandler("logs/optimization_results.log")
            file_formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def prepare_data(self, df, timeframe, params):
        try:
            ohlcv_data = self.processor.resample_to_ohlcv(df, timeframe=timeframe)
            df = ohlcv_data.reset_index()
            
            df = add_bollinger_bands(df, window=params['bb_window'], num_std=params['bb_std'])
            df = add_rsi(df, period=params['rsi_period'])
            df = add_atr(df, period=params['atr_period'])
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                            'upper_band', 'middle_band', 'lower_band', 'rsi', 'atr']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return pd.DataFrame()
    
    def objective(self, trial):
        try:
            timeframe = trial.suggest_categorical('timeframe', ['5min', '15min', '30min', '1h', '4h', '1d'])
            bb_window = trial.suggest_int('bb_window', 10, 50)
            bb_std = trial.suggest_float('bb_std', 1.0, 3.0, step=0.1)
            rsi_period = trial.suggest_int('rsi_period', 5, 30)
            rsi_lower = trial.suggest_int('rsi_lower', 20, 40)
            rsi_upper = trial.suggest_int('rsi_upper', 60, 80)
            atr_period = trial.suggest_int('atr_period', 5, 30)
            take_profit_mult = trial.suggest_float('take_profit_mult', 1.0, 6.0, step=0.2)
            stop_loss_mult = trial.suggest_float('stop_loss_mult', 0.5, 2.0, step=0.1)
            
            params = {
                'bb_window': bb_window,
                'bb_std': bb_std,
                'rsi_period': rsi_period,
                'rsi_lower': rsi_lower,
                'rsi_upper': rsi_upper,
                'atr_period': atr_period,
                'take_profit_mult': take_profit_mult,
                'stop_loss_mult': stop_loss_mult
            }
            
            train_df = self.prepare_data(self.train_data, timeframe, params)
            
            if train_df.empty:
                self.logger.warning(f"Trial {trial.number}: Empty dataframe after preparation")
                return -1.0
            
            signal_gen = SignalGenerator(rsi_lower=rsi_lower, rsi_upper=rsi_upper)
            signals_df = signal_gen.generate_signals(train_df)
            
            backtest_results = self.backtest_engine.run_backtest(signals_df, self.risk_manager)
            
            if 'trades' not in backtest_results or backtest_results['trades'].empty:
                self.logger.info(f"Trial {trial.number}, "
                            f"Timeframe: {timeframe}, "
                            f"BB[{bb_window}, {bb_std}], "
                            f"RSI[{rsi_period}, {rsi_lower}, {rsi_upper}], "
                            f"ATR[{atr_period}], "
                            f"TP/SL[{take_profit_mult}/{stop_loss_mult}], "
                            f"Trades: 0, "
                            f"Score: -1.0000")
                return -1.0
            
            performance = PerformanceMetrics(
                backtest_results['trades'],
                backtest_results['portfolio_history']
            )
            
            metrics = performance.generate_report()
            sharpe_ratio = metrics['Sharpe Ratio']
            total_trades = metrics['Total Trades']
            win_rate = metrics['Win Rate']
            max_drawdown = abs(metrics['Maximum Drawdown'])
            
            if total_trades < 10:
                score = -1.0
            else:
                score = sharpe_ratio - (0.1 * max_drawdown) + (total_trades / 1000)
                
                if sharpe_ratio <= 0:
                    score = performance.total_return() * 10
            
            self.logger.info(f"Trial {trial.number}, "
                            f"Timeframe: {timeframe}, "
                            f"BB[{bb_window}, {bb_std}], "
                            f"RSI[{rsi_period}, {rsi_lower}, {rsi_upper}], "
                            f"ATR[{atr_period}], "
                            f"TP/SL[{take_profit_mult}/{stop_loss_mult}], "
                            f"Trades: {total_trades}, "
                            f"Win Rate: {win_rate:.2f}, "
                            f"Sharpe: {sharpe_ratio:.2f}, "
                            f"MDD: {max_drawdown:.2f}, "
                            f"Score: {score:.4f}")
            
            return score
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return -1.0
    
    def optimize(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"\nBest parameters: {best_params}")
        self.logger.info(f"Best value: {best_value}")
        
        return best_params
        
    def validate_best_parameters(self, best_params):
        if self.test_data is None or self.test_data.empty:
            self.logger.warning("No test data provided for validation")
            return None, None
        
        try:
            timeframe = best_params.pop('timeframe', '15min')
            
            params = {
                'bb_window': best_params.get('bb_window', 20),
                'bb_std': best_params.get('bb_std', 1.8),
                'rsi_period': best_params.get('rsi_period', 13),
                'rsi_lower': best_params.get('rsi_lower', 30),
                'rsi_upper': best_params.get('rsi_upper', 70),
                'atr_period': best_params.get('atr_period', 14),
                'take_profit_mult': best_params.get('take_profit_mult', 4.0),
                'stop_loss_mult': best_params.get('stop_loss_mult', 1.0)
            }
            
            test_df = self.prepare_data(self.test_data, timeframe, params)
            
            if test_df.empty:
                self.logger.warning("Empty test dataframe after preparation")
                return None, None
            
            signal_gen = SignalGenerator(
                rsi_lower=params['rsi_lower'],
                rsi_upper=params['rsi_upper']
            )
            signals_df = signal_gen.generate_signals(test_df)
            
            backtest_results = self.backtest_engine.run_backtest(signals_df, self.risk_manager)
            
            performance = PerformanceMetrics(
                backtest_results['trades'],
                backtest_results['portfolio_history']
            )
            
            metrics = performance.generate_report()
            self.logger.info("\nOut-of-sample validation results:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value}")
                
            return backtest_results, metrics
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return None, None