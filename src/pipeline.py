import os
import logging
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.indicators.bollinger_bands import add_bollinger_bands
from src.indicators.rsi import add_rsi
from src.indicators.atr import add_atr
from src.strategy.signal_generator import SignalGenerator
from src.strategy.risk_manager import RiskManager
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.performance import PerformanceMetrics
from src.visualization.backtest import plot_backtest_results

class TradingPipeline:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader()
        self.processor = DataProcessor()
        self.risk_manager = RiskManager(
            max_daily_loss=config.get('risk_management', {}).get('max_daily_loss', 2.0),
            max_drawdown=config.get('risk_management', {}).get('max_drawdown', 0.05),
            risk_per_trade=config.get('risk_management', {}).get('risk_per_trade', 0.005)
        )
        self.backtest_engine = BacktestEngine(
            initial_balance=config.get('backtest', {}).get('initial_capital', 100000),
            commission=config.get('backtest', {}).get('commission', 0.001)
        )
        self.signal_generator = SignalGenerator(
            rsi_lower=config.get('parameters', {}).get('rsi_lower', 30),
            rsi_upper=config.get('parameters', {}).get('rsi_upper', 70),
            trading_start=config.get('parameters', {}).get('trading_start', '09:15'),
            trading_end=config.get('parameters', {}).get('trading_end', '14:30')
        )
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger("trading_pipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            if 'save_dir' in self.config:
                os.makedirs(self.config['save_dir'], exist_ok=True)
                file_handler = logging.FileHandler(os.path.join(self.config['save_dir'], 'backtest.log'))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
        return logger
    
    def prepare_data(self, data, timeframe='15min'):
        self.logger.info(f"Resampling data to {timeframe} timeframe")
        ohlcv_data = self.processor.resample_to_ohlcv(data, timeframe=timeframe)
        
        df = ohlcv_data.reset_index()
        
        self.logger.info("Adding indicators to data")
        bb_window = self.config.get('parameters', {}).get('bb_window', 20)
        bb_std = self.config.get('parameters', {}).get('bb_std', 1.8)
        rsi_period = self.config.get('parameters', {}).get('rsi_period', 13)
        atr_period = self.config.get('parameters', {}).get('atr_period', 14)
        
        df = add_bollinger_bands(df, window=bb_window, num_std=bb_std)
        df = add_rsi(df, period=rsi_period)
        df = add_atr(df, period=atr_period)
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'upper_band', 'middle_band', 'lower_band', 'rsi', 'atr']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def run_backtest(self, start_date, end_date, timeframe='15min'):
        self.logger.info(f"Starting backtest from {start_date} to {end_date} with {timeframe} timeframe")
        
        self.logger.info("Loading market data")
        data = self.data_loader.get_active_contract_data(start_date, end_date)
        
        if data.empty:
            self.logger.error("No data found for the specified date range")
            return None, None
            
        df = self.prepare_data(data, timeframe)
        
        self.logger.info("Generating trading signals")
        signals_df = self.signal_generator.generate_signals(df)
        
        self.logger.info("Running backtest simulation")
        results = self.backtest_engine.run_backtest(signals_df, self.risk_manager)
        
        self.logger.info("Calculating performance metrics")
        performance = PerformanceMetrics(
            results['trades'],
            results['portfolio_history'],
            risk_free_rate=self.config.get('backtest', {}).get('risk_free_rate', 0.03)
        )
        
        metrics = performance.generate_report()
        
        self.logger.info("Backtest completed. Summary of results:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value}")
            
        return results, signals_df
    
    def visualize_results(self, results, signals_df, save_path=None):
        if results is None or signals_df is None:
            self.logger.error("No results to visualize")
            return
            
        self.logger.info("Generating visualization of backtest results")
        
        fig = plot_backtest_results(signals_df, results['trades'])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            self.logger.info(f"Visualization saved to {save_path}")
            
        return fig