import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import numpy as np
from .base import save_figure, setup_plot_style

def plot_price_with_indicators(df, ticker=None, window=None, figsize=(16, 12), 
                              bb_columns=None, rsi_column=None):
    if bb_columns is None:
        bb_columns = ('upper_band', 'middle_band', 'lower_band')
    
    if rsi_column is None:
        rsi_column = 'rsi'
    
    has_bb = all(col in df.columns for col in bb_columns)
    has_rsi = rsi_column in df.columns
    
    if window is not None and window < len(df):
        plot_df = df.iloc[-window:].copy()
    else:
        plot_df = df.copy()
    
    if isinstance(plot_df.index, pd.DatetimeIndex) and 'datetime' not in plot_df.columns:
        plot_df = plot_df.reset_index()
    
    fig = plt.figure(figsize=figsize)
    
    if has_rsi:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
    else:
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = None
    
    ax1.plot(plot_df['datetime'], plot_df['close'], label='Close Price', color='black', linewidth=1.5)
    
    if has_bb:
        upper_band, middle_band, lower_band = bb_columns
        ax1.plot(plot_df['datetime'], plot_df[middle_band], label='SMA(20)', color='blue', alpha=0.7)
        ax1.plot(plot_df['datetime'], plot_df[upper_band], label='Upper BB', color='red', linestyle='--', alpha=0.7)
        ax1.plot(plot_df['datetime'], plot_df[lower_band], label='Lower BB', color='green', linestyle='--', alpha=0.7)
        
        ax1.fill_between(plot_df['datetime'], plot_df[upper_band], plot_df[lower_band], 
                         color='gray', alpha=0.1)
    
    ticker_text = f" - {ticker}" if ticker else ""
    ax1.set_title(f'Price with Indicators{ticker_text}', fontsize=16)
    ax1.set_ylabel('Price', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    if has_rsi and ax2 is not None:
        ax2.plot(plot_df['datetime'], plot_df[rsi_column], label='RSI(13)', color='purple', linewidth=1.5)
        
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        
        ax2.fill_between(plot_df['datetime'], 70, 100, color='red', alpha=0.1)
        ax2.fill_between(plot_df['datetime'], 0, 30, color='green', alpha=0.1)
        
        ax2.set_ylabel('RSI', fontsize=14)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    if ax2 is not None:
        ax2.set_xlabel('Date', fontsize=14)
        plt.setp(ax1.get_xticklabels(), visible=False)
    else:
        ax1.set_xlabel('Date', fontsize=14)
    
    plt.tight_layout()
    return fig

def plot_backtest_results(df, trades_df=None, indicators=True, figsize=(18, 14), save_path=None):
    setup_plot_style()
    
    if isinstance(df.index, pd.DatetimeIndex) and 'datetime' not in df.columns:
        df = df.reset_index()
    
    has_bb = all(col in df.columns for col in ['upper_band', 'middle_band', 'lower_band'])
    has_rsi = 'rsi' in df.columns
    has_trades = trades_df is not None and not trades_df.empty
    
    fig = plt.figure(figsize=figsize)
    
    if has_rsi:
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = None
        ax3 = fig.add_subplot(gs[1], sharex=ax1)
    
    ax1.plot(df['datetime'], df['close'], label='Close Price', color='black', linewidth=1.5)
    
    if has_bb and indicators:
        ax1.plot(df['datetime'], df['middle_band'], label='SMA(20)', color='blue', alpha=0.7)
        ax1.plot(df['datetime'], df['upper_band'], label='Upper BB', color='red', linestyle='--', alpha=0.7)
        ax1.plot(df['datetime'], df['lower_band'], label='Lower BB', color='green', linestyle='--', alpha=0.7)
        
        ax1.fill_between(df['datetime'], df['upper_band'], df['lower_band'], 
                        color='gray', alpha=0.1)
    
    if has_trades:
        buy_trades = trades_df[trades_df['type'] == 'buy']
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        if not buy_trades.empty:
            ax1.scatter(buy_trades['entry_time'], buy_trades['entry_price'], 
                       marker='^', color='green', s=100, label='Buy')
            ax1.scatter(buy_trades['exit_time'], buy_trades['exit_price'], 
                       marker='v', color='red', s=100, label='Exit Buy')
        
        if not sell_trades.empty:
            ax1.scatter(sell_trades['entry_time'], sell_trades['entry_price'], 
                       marker='v', color='red', s=100, label='Sell')
            ax1.scatter(sell_trades['exit_time'], sell_trades['exit_price'], 
                       marker='^', color='green', s=100, label='Exit Sell')
        
        for _, trade in trades_df.iterrows():
            ax1.plot([trade['entry_time'], trade['exit_time']], 
                    [trade['entry_price'], trade['exit_price']], 
                    color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_title('Backtest Results', fontsize=16)
    ax1.set_ylabel('Price', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    if has_rsi and indicators and ax2 is not None:
        ax2.plot(df['datetime'], df['rsi'], label='RSI(13)', color='purple', linewidth=1.5)
        
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        
        ax2.fill_between(df['datetime'], 70, 100, color='red', alpha=0.1)
        ax2.fill_between(df['datetime'], 0, 30, color='green', alpha=0.1)
        
        ax2.set_ylabel('RSI', fontsize=14)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    if has_trades:
        trades_df = trades_df.sort_values('entry_time')
        trades_df['profit'] = trades_df['exit_price'] - trades_df['entry_price']
        trades_df.loc[trades_df['type'] == 'sell', 'profit'] = -trades_df.loc[trades_df['type'] == 'sell', 'profit']
        trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
        
        ax3.plot(trades_df['exit_time'], trades_df['cumulative_profit'], 
                color='blue', linewidth=2, label='Equity Curve')
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax3.set_ylabel('Cumulative Profit', fontsize=14)
        ax3.set_xlabel('Date', fontsize=14)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.set_visible(False)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    if ax2 is not None:
        plt.setp(ax1.get_xticklabels(), visible=False)
        if ax3 is not None:
            plt.setp(ax2.get_xticklabels(), visible=False)
    
    plt.tight_layout()
    save_figure(fig, save_path)
    return fig

def plot_candles(df, ticker=None, window=None, figsize=(16, 8), volume=True, save_path=None):
    setup_plot_style()
    
    try:
        import mplfinance as mpf
    except ImportError:
        print("Installing mplfinance...")
        import pip
        pip.main(['install', 'mplfinance'])
        import mplfinance as mpf
    
    if window is not None and window < len(df):
        plot_df = df.iloc[-window:].copy()
    else:
        plot_df = df.copy()
    
    if isinstance(plot_df.index, pd.DatetimeIndex) and 'datetime' not in plot_df.columns:
        plot_df_copy = plot_df.copy()
    else:
        plot_df_copy = plot_df.copy()
        plot_df_copy.set_index('datetime', inplace=True)
    
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in plot_df_copy.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    mc = mpf.make_marketcolors(
        up='forestgreen', down='crimson',
        edge='inherit',
        wick={'up': 'limegreen', 'down': 'tomato'},
        volume={'up': 'forestgreen', 'down': 'crimson'},
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        y_on_right=False,
        facecolor='white',
        figcolor='white',
        gridcolor='lightgray'
    )
    
    fig, axes = mpf.plot(
        plot_df_copy,
        type='candle',
        style=s,
        title=f'Candlestick Chart{" - " + ticker if ticker else ""}',
        ylabel='Price',
        volume=volume,
        figsize=figsize,
        returnfig=True
    )
    
    if save_path:
        fig.savefig(save_path)
    
    return fig

def plot_performance_metrics(metrics_dict, key_metrics=None, figsize=(12, 6), save_path=None):
    setup_plot_style()
    
    if key_metrics is None:
        key_metrics = ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Maximum Drawdown', 'Total Return']
    
    metrics_df = pd.DataFrame({
        'Metric': list(metrics_dict.keys()),
        'Value': list(metrics_dict.values())
    })
    
    key_metrics_df = metrics_df[metrics_df['Metric'].isin(key_metrics)]
    
    fig = plt.figure(figsize=figsize)
    ax = sns.barplot(x='Metric', y='Value', data=key_metrics_df)
    plt.title('Key Performance Metrics', fontsize=16)
    plt.ylabel('Value')
    plt.xticks(rotation=0)
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.4f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    save_figure(fig, save_path)
    return fig

def plot_equity_curve(portfolio_history, initial_capital=100000, figsize=(14, 7), save_path=None):
    setup_plot_style()
    
    if len(portfolio_history) <= 1:
        return None
    
    fig = plt.figure(figsize=figsize)
    plt.plot(portfolio_history, label='Portfolio Value')
    plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
    plt.title('Portfolio Equity Curve', fontsize=16)
    plt.xlabel('Trade Number')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig

def plot_trade_analysis(trades_df, figsize_trade=(14, 7), figsize_exit=(10, 6), save_path=None):
    setup_plot_style()
    
    if trades_df.empty:
        return None, None, None
    
    fig_trades = plt.figure(figsize=figsize_trade)
    
    colors = ['green' if profit > 0 else 'red' for profit in trades_df['profit']]
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(trades_df)), trades_df['profit'], color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Trade Results', fontsize=14)
    plt.xlabel('Trade Number')
    plt.ylabel('Profit/Loss')
    
    plt.subplot(1, 2, 2)
    sns.histplot(trades_df['profit'], bins=15, kde=True)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    plt.title('Profit/Loss Distribution', fontsize=14)
    plt.xlabel('Profit/Loss')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    fig_exit = None
    profit_by_exit = None
    
    if 'exit_reason' in trades_df.columns:
        fig_exit = plt.figure(figsize=figsize_exit)
        exit_counts = trades_df['exit_reason'].value_counts()
        ax = exit_counts.plot(kind='bar', color='skyblue')
        plt.title('Trade Exit Reasons', fontsize=14)
        plt.xlabel('Exit Reason')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        for i, v in enumerate(exit_counts):
            ax.text(i, v + 0.1, str(v), ha='center')
            
        plt.tight_layout()
        
        profit_by_exit = trades_df.groupby('exit_reason')['profit'].agg(['mean', 'sum', 'count'])
    
    if save_path:
        fig_trades.savefig(f"{save_path}_trades.png")
        if fig_exit:
            fig_exit.savefig(f"{save_path}_exit_reasons.png")
    
    return fig_trades, fig_exit, profit_by_exit