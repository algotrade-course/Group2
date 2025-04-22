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

os.makedirs('../results', exist_ok=True)
os.makedirs('../data_cache', exist_ok=True)
os.makedirs('../logs', exist_ok=True)

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

from data.data_loader import DataLoader
from optimization.config_loader import ConfigLoader
from optimization.strategy_optimizer import StrategyOptimizer
from optimization.optimization_analyzer import OptimizationAnalyzer
from visualization.optimization import plot_optimization_progress, plot_parameter_importance, plot_timeframe_analysis
from visualization.comparison import plot_parameter_comparison, plot_equity_curves_comparison
from pipeline import TradingPipeline
from backtest.performance import PerformanceMetrics

def run_optimization(train_start_date='2023-01-01', train_end_date='2023-01-30', 
                    test_start_date='2023-07-01', test_end_date='2023-07-31', 
                    n_trials=100, save_plots=True):
    
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    
    optimization_config = config.get("optimization", {})
    timeframes = optimization_config.get("timeframes", ["5min", "15min", "30min", "1h", "4h"])
    param_ranges = optimization_config.get("parameter_ranges", {})
    
    print(f"Training period: {train_start_date} to {train_end_date}")
    print(f"Validation period: {test_start_date} to {test_end_date}")
    print(f"Number of optimization trials: {n_trials}")
    
    loader = DataLoader(cache_dir="../data_cache")
    
    print("Loading training data...")
    train_data = loader.get_active_contract_data(train_start_date, train_end_date)
    print(f"Loaded {len(train_data)} tick data points for training period")
    
    print("\nLoading validation data...")
    test_data = loader.get_active_contract_data(test_start_date, test_end_date)
    print(f"Loaded {len(test_data)} tick data points for validation period")
    
    if train_data.empty or test_data.empty:
        raise ValueError("Failed to load market data. Please check database connection and date range.")
    
    optimizer = StrategyOptimizer(
        train_data=train_data,
        test_data=test_data,
        initial_capital=100000,
        n_trials=n_trials
    )
    
    print(f"Starting optimization with {n_trials} trials...")
    best_params = optimizer.optimize()
    
    print("\nOptimization complete!")
    print(f"Best parameters: {best_params}")
    
    analyzer = OptimizationAnalyzer(log_file="logs/optimization_results.log")
    opt_df = analyzer.trials_df
    print(f"Found {len(opt_df)} optimization trials")
    
    if save_plots:
        fig_progress = analyzer.plot_optimization_progress(save_path='../results/optimization_progress.png')
        plt.close(fig_progress)
        
        fig_importance = analyzer.plot_parameter_importance(save_path='../results/parameter_importance.png')
        plt.close(fig_importance)
        
        fig_timeframe, timeframe_analysis = analyzer.plot_timeframe_analysis(save_dir='../results')
        plt.close(fig_timeframe)
        
        print("Performance by Timeframe:")
        print(timeframe_analysis)
    
    top_params = analyzer.get_top_parameters(n=5)
    print("Top 5 parameter combinations:")
    print(top_params)
    top_params.to_csv('../results/top_parameters.csv', index=False)
    
    summary = analyzer.generate_summary_report(save_dir='../results')
    best_params = summary['best_parameters']
    
    print("Best parameters from optimization:")
    for param, value in best_params.items():
        print(f"- {param}: {value}")
    
    validation_results, metrics = optimizer.validate_best_parameters(best_params)
    
    print("\nOut-of-Sample Validation Results:")
    if metrics is not None:
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, (int, float)) else f"{metric}: {value}")
    else:
        print("Validation failed or no trades were executed")
    
    default_params = config['parameters'].copy()
    print("Running backtest with default parameters...")
    
    pipeline_default = TradingPipeline(config)
    default_results, default_signals_df = pipeline_default.run_backtest(test_start_date, test_end_date, default_params['default_timeframe'])
    
    if default_results and 'trades' in default_results and not default_results['trades'].empty:
        default_performance = PerformanceMetrics(
            default_results['trades'],
            default_results['portfolio_history']
        )
        default_metrics = default_performance.generate_report()
        
        print("Performance with Default Parameters:")
        for metric, value in default_metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, (int, float)) else f"{metric}: {value}")
    else:
        default_metrics = None
        print("No trades were executed with default parameters")
    
    if default_metrics and metrics and save_plots:
        fig_comparison, comparison_df = plot_parameter_comparison(
            default_metrics, 
            metrics, 
            save_path='../results/metrics_comparison.png'
        )
        plt.close(fig_comparison)
        
        print("Comparison of Default vs Optimized Parameters:")
        print(comparison_df)
        comparison_df.to_csv('../results/parameter_comparison.csv')
        
        if 'portfolio_history' in default_results and 'portfolio_history' in validation_results:
            fig_curves = plot_equity_curves_comparison(
                default_results['portfolio_history'],
                validation_results['portfolio_history'],
                save_path='../results/default_vs_optimized.png'
            )
            plt.close(fig_curves)
    else:
        print("Cannot compare metrics - insufficient data")
    
    best_trial = analyzer.get_top_parameters(n=1).iloc[0].to_dict()
    
    optimized_config = config.copy()
    
    optimized_config['parameters']['bb_window'] = best_params['bb_window']
    optimized_config['parameters']['bb_std'] = best_params['bb_std']
    optimized_config['parameters']['rsi_period'] = best_params['rsi_period']
    optimized_config['parameters']['rsi_lower'] = best_params['rsi_lower']
    optimized_config['parameters']['rsi_upper'] = best_params['rsi_upper']
    optimized_config['parameters']['atr_period'] = best_params['atr_period']
    optimized_config['parameters']['take_profit_mult'] = best_params['take_profit_mult']
    optimized_config['parameters']['stop_loss_mult'] = best_params['stop_loss_mult']
    
    if 'timeframe' in best_params:
        optimized_config['parameters']['default_timeframe'] = best_params['timeframe']
    else:
        optimized_config['parameters']['default_timeframe'] = best_trial.get('timeframe', config['parameters']['default_timeframe'])
    
    optimized_config['optimization_results'] = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'train_period': f'{train_start_date} to {train_end_date}',
        'test_period': f'{test_start_date} to {test_end_date}',
        'trials': n_trials,
        'best_score': float(best_trial['score']) if 'score' in best_trial else float(best_params.get('score', 0))
    }
    
    if metrics:
        optimized_config['performance'] = {
            metric: float(value) if isinstance(value, (int, float, np.number)) else value 
            for metric, value in metrics.items()
            if metric in ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Maximum Drawdown', 'Total Return', 'Total Trades']
        }
    
    output_path = '../config/optimized_parameters.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(optimized_config, f, indent=4)
    
    print(f'Optimized parameters saved to {output_path}')
    
    optimized_params = {
        'parameters': best_params,
        'optimization_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'train_period': f'{train_start_date} to {train_end_date}',
            'test_period': f'{test_start_date} to {test_end_date}',
            'best_score': float(best_trial['score']) if 'score' in best_trial else float(best_params.get('score', 0))
        },
        'performance': optimized_config.get('performance', {})
    }
    
    with open('../results/optimized_parameters.json', 'w') as f:
        json.dump(optimized_params, f, indent=4)
        
    print('Optimized parameters also saved to results/optimized_parameters.json')
    
    return best_params, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run trading strategy optimization')
    parser.add_argument('--train-start', type=str, default='2024-01-01', help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, default='2024-06-01', help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', type=str, default='2024-06-01', help='Testing start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, default='2025-01-01', help='Testing end date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    
    args = parser.parse_args()
    
    best_params, metrics = run_optimization(
        train_start_date=args.train_start,
        train_end_date=args.train_end,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        n_trials=args.trials,
        save_plots=not args.no_plots
    )