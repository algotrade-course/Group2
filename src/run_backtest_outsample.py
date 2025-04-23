import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'outsample')
DATA_CACHE_DIR = os.path.join(PROJECT_ROOT, 'data_cache')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

from src.data.data_loader import DataLoader
from src.optimization.config_loader import ConfigLoader
from src.pipeline import TradingPipeline
from src.backtest.performance import PerformanceMetrics
from src.visualization.comparison import plot_parameter_comparison, plot_equity_curves_comparison
from src.visualization.backtest import plot_backtest_results, plot_equity_curve, plot_trade_analysis

def run_outsample_backtest(test_start_date='2024-06-01', test_end_date='2025-01-01', 
                          save_plots=True, use_reference=True, optimized_config_path=None):
    
    config_loader = ConfigLoader()
    default_config = config_loader.get_config()
    
    if use_reference:
        reference_config_path = os.path.join(CONFIG_DIR, 'reference_optimized_parameters.json')
        if not os.path.exists(reference_config_path):
            print(f"Reference optimized parameters file not found at {reference_config_path}")
            print("Using latest optimized parameters instead.")
            use_reference = False
    
    if not use_reference:
        if optimized_config_path is None:
            optimized_config_path = os.path.join(CONFIG_DIR, 'optimized_parameters.json')
        
        if not os.path.exists(optimized_config_path):
            raise FileNotFoundError(f"Optimized parameters file not found at {optimized_config_path}")
    else:
        optimized_config_path = reference_config_path
    
    with open(optimized_config_path, 'r') as f:
        optimized_config = json.load(f)
    
    print(f"Using {'reference' if use_reference else 'custom'} optimized parameters from: {optimized_config_path}")
    print(f"Out-of-sample testing period: {test_start_date} to {test_end_date}")
    
    optimized_params = optimized_config['parameters']
    
    print("\nOptimized Parameters:")
    for param, value in optimized_params.items():
        if param not in ['trailing_trigger', 'trailing_atr', 'trading_start', 'trading_end', 'market_close_time']:
            print(f"- {param}: {value}")
    
    default_params = default_config['parameters']
    
    print("\nDefault Parameters:")
    for param, value in default_params.items():
        if param not in ['trailing_trigger', 'trailing_atr', 'trading_start', 'trading_end', 'market_close_time']:
            print(f"- {param}: {value}")
    
    print("\nRunning backtest with optimized parameters...")
    
    temp_optimized_config = default_config.copy()
    temp_optimized_config['parameters'] = optimized_params
    
    optimized_pipeline = TradingPipeline(temp_optimized_config)
    
    optimized_timeframe = optimized_params.get('default_timeframe', '15min')
    
    optimized_results, optimized_signals_df = optimized_pipeline.run_backtest(
        test_start_date, test_end_date, optimized_timeframe)
    
    if optimized_results and 'trades' in optimized_results and not optimized_results['trades'].empty:
        optimized_performance = PerformanceMetrics(
            optimized_results['trades'],
            optimized_results['portfolio_history'],
            risk_free_rate=default_config.get('backtest', {}).get('risk_free_rate', 0.03)
        )
        optimized_metrics = optimized_performance.generate_report()
        
        print("\nPerformance with Optimized Parameters:")
        for metric, value in optimized_metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, (int, float)) else f"{metric}: {value}")
        
        if save_plots:
            fig = plot_backtest_results(
                optimized_signals_df,
                trades_df=optimized_results['trades'],
                save_path=os.path.join(RESULTS_DIR, 'optimized_backtest_chart.png')
            )
            plt.close(fig)
            
            fig = plot_equity_curve(
                optimized_results['portfolio_history'],
                save_path=os.path.join(RESULTS_DIR, 'optimized_equity_curve.png')
            )
            plt.close(fig)
            
            fig_trades, fig_exit, profit_by_exit = plot_trade_analysis(
                optimized_results['trades'],
                save_path=os.path.join(RESULTS_DIR, 'optimized_trade_analysis')
            )
            
            if fig_trades:
                plt.close(fig_trades)
            
            if fig_exit:
                plt.close(fig_exit)
    else:
        optimized_metrics = None
        print("No trades were executed with optimized parameters")
    
    print("\nRunning backtest with default parameters for comparison...")
    
    default_pipeline = TradingPipeline(default_config)
    default_timeframe = default_params.get('default_timeframe', '15min')
    
    default_results, default_signals_df = default_pipeline.run_backtest(
        test_start_date, test_end_date, default_timeframe)
    
    if default_results and 'trades' in default_results and not default_results['trades'].empty:
        default_performance = PerformanceMetrics(
            default_results['trades'],
            default_results['portfolio_history'],
            risk_free_rate=default_config.get('backtest', {}).get('risk_free_rate', 0.03)
        )
        default_metrics = default_performance.generate_report()
        
        print("\nPerformance with Default Parameters:")
        for metric, value in default_metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, (int, float)) else f"{metric}: {value}")
        
        if save_plots:
            fig = plot_backtest_results(
                default_signals_df,
                trades_df=default_results['trades'],
                save_path=os.path.join(RESULTS_DIR, 'default_backtest_chart.png')
            )
            plt.close(fig)
            
            fig = plot_equity_curve(
                default_results['portfolio_history'],
                save_path=os.path.join(RESULTS_DIR, 'default_equity_curve.png')
            )
            plt.close(fig)
    else:
        default_metrics = None
        print("No trades were executed with default parameters")
    
    if default_metrics and optimized_metrics and save_plots:
        print("\nGenerating comparison visualizations...")
        
        fig_comparison, comparison_df = plot_parameter_comparison(
            default_metrics, 
            optimized_metrics, 
            save_path=os.path.join(RESULTS_DIR, 'metrics_comparison.png')
        )
        plt.close(fig_comparison)
        
        print("\nComparison of Default vs Optimized Parameters:")
        print(comparison_df)
        comparison_df.to_csv(os.path.join(RESULTS_DIR, 'parameter_comparison.csv'))
        
        if 'portfolio_history' in default_results and 'portfolio_history' in optimized_results:
            fig_curves = plot_equity_curves_comparison(
                default_results['portfolio_history'],
                optimized_results['portfolio_history'],
                save_path=os.path.join(RESULTS_DIR, 'default_vs_optimized.png')
            )
            plt.close(fig_curves)
    else:
        print("Cannot compare metrics - insufficient data")
    
    if optimized_metrics:
        result_data = {
            "test_period": {
                "start_date": test_start_date,
                "end_date": test_end_date
            },
            "optimized_source": "reference" if use_reference else "custom",
            "optimized_parameters": optimized_params,
            "optimized_metrics": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in optimized_metrics.items()
            },
            "default_metrics": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in default_metrics.items()
            } if default_metrics else None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(RESULTS_DIR, 'outsample_backtest_results.json'), 'w') as f:
            json.dump(result_data, f, indent=4)
        
        print(f"\nOut-of-sample backtest results saved to {RESULTS_DIR}/outsample_backtest_results.json")
    
    return optimized_results, optimized_metrics, default_results, default_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run out-of-sample backtest with optimized parameters')
    parser.add_argument('--start', type=str, default='2024-06-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--custom', action='store_true', help='Use custom optimized parameters instead of reference')
    parser.add_argument('--config', type=str, default=None, help='Path to custom optimized parameters config file')
    
    args = parser.parse_args()

    use_reference = not args.custom
    
    optimized_results, optimized_metrics, default_results, default_metrics = run_outsample_backtest(
        test_start_date=args.start,
        test_end_date=args.end,
        save_plots=not args.no_plots,
        use_reference=use_reference,
        optimized_config_path=args.config
    )