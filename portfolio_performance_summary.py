import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_key_statistics(portfolio_returns, benchmark_returns, portfolio_weights_history=None):
    """
    Calculate key portfolio statistics including tracking error, information ratio, 
    drawdown, average holdings, and turnover
    
    Parameters:
    portfolio_returns: Series of portfolio returns
    benchmark_returns: Series of benchmark returns
    portfolio_weights_history: DataFrame with historical portfolio weights (optional for turnover)
    """
    
    # Align returns
    aligned_data = pd.DataFrame({
        'Portfolio': portfolio_returns,
        'Benchmark': benchmark_returns
    }).dropna()
    
    portfolio_ret = aligned_data['Portfolio']
    benchmark_ret = aligned_data['Benchmark']
    excess_returns = portfolio_ret - benchmark_ret
    
    # 1. Annualized Tracking Error
    tracking_error = excess_returns.std() * np.sqrt(252)
    
    # 2. Information Ratio
    excess_annual_return = excess_returns.mean() * 252
    information_ratio = excess_annual_return / tracking_error if tracking_error > 0 else 0
    
    # 3. Maximum Drawdown
    def calculate_max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    portfolio_max_drawdown = calculate_max_drawdown(portfolio_ret)
    benchmark_max_drawdown = calculate_max_drawdown(benchmark_ret)
    
    # 4. Average Holdings (if weights history provided)
    if portfolio_weights_history is not None:
        # Calculate average number of holdings over time
        avg_holdings = (portfolio_weights_history > 0.001).sum(axis=1).mean()  # Holdings > 0.1%
        
        # 5. Portfolio Turnover
        # Turnover = average of absolute weight changes
        weight_changes = portfolio_weights_history.diff().abs()
        turnover = weight_changes.sum(axis=1).mean() * 12  # Annualized (assuming monthly rebalancing)
    else:
        avg_holdings = np.nan
        turnover = np.nan
    
    # Additional statistics
    portfolio_annual_return = portfolio_ret.mean() * 252
    portfolio_annual_vol = portfolio_ret.std() * np.sqrt(252)
    benchmark_annual_return = benchmark_ret.mean() * 252
    benchmark_annual_vol = benchmark_ret.std() * np.sqrt(252)
    
    portfolio_sharpe = portfolio_annual_return / portfolio_annual_vol if portfolio_annual_vol > 0 else 0
    benchmark_sharpe = benchmark_annual_return / benchmark_annual_vol if benchmark_annual_vol > 0 else 0
    
    # Win rate
    win_rate = (excess_returns > 0).mean() * 100
    
    key_stats = {
        'Portfolio_Annual_Return': portfolio_annual_return,
        'Benchmark_Annual_Return': benchmark_annual_return,
        'Excess_Return': excess_annual_return,
        'Portfolio_Volatility': portfolio_annual_vol,
        'Benchmark_Volatility': benchmark_annual_vol,
        'Tracking_Error': tracking_error,
        'Information_Ratio': information_ratio,
        'Portfolio_Sharpe': portfolio_sharpe,
        'Benchmark_Sharpe': benchmark_sharpe,
        'Portfolio_Max_Drawdown': portfolio_max_drawdown,
        'Benchmark_Max_Drawdown': benchmark_max_drawdown,
        'Average_Holdings': avg_holdings,
        'Annual_Turnover': turnover,
        'Win_Rate': win_rate
    }
    
    return key_stats, aligned_data

def print_key_statistics_table(key_stats):
    """
    Print formatted table of key portfolio statistics
    """
    
    print("\n" + "="*70)
    print("KEY PORTFOLIO STATISTICS")
    print("="*70)
    
    print(f"{'RETURNS':<25}")
    print(f"  Portfolio Return      : {key_stats['Portfolio_Annual_Return']:>8.2%}")
    print(f"  Benchmark Return      : {key_stats['Benchmark_Annual_Return']:>8.2%}")
    print(f"  Excess Return         : {key_stats['Excess_Return']:>8.2%}")
    print()
    
    print(f"{'RISK METRICS':<25}")
    print(f"  Portfolio Volatility  : {key_stats['Portfolio_Volatility']:>8.2%}")
    print(f"  Benchmark Volatility  : {key_stats['Benchmark_Volatility']:>8.2%}")
    print(f"  Tracking Error        : {key_stats['Tracking_Error']:>8.2%}")
    print()
    
    print(f"{'RISK-ADJUSTED METRICS':<25}")
    print(f"  Information Ratio     : {key_stats['Information_Ratio']:>8.3f}")
    print(f"  Portfolio Sharpe      : {key_stats['Portfolio_Sharpe']:>8.3f}")
    print(f"  Benchmark Sharpe      : {key_stats['Benchmark_Sharpe']:>8.3f}")
    print()
    
    print(f"{'DRAWDOWN ANALYSIS':<25}")
    print(f"  Portfolio Max DD      : {key_stats['Portfolio_Max_Drawdown']:>8.2%}")
    print(f"  Benchmark Max DD      : {key_stats['Benchmark_Max_Drawdown']:>8.2%}")
    print()
    
    print(f"{'PORTFOLIO CHARACTERISTICS':<25}")
    if not np.isnan(key_stats['Average_Holdings']):
        print(f"  Average Holdings      : {key_stats['Average_Holdings']:>8.1f}")
    else:
        print(f"  Average Holdings      : {'N/A':>8}")
    
    if not np.isnan(key_stats['Annual_Turnover']):
        print(f"  Annual Turnover       : {key_stats['Annual_Turnover']:>8.1%}")
    else:
        print(f"  Annual Turnover       : {'N/A':>8}")
    
    print(f"  Win Rate              : {key_stats['Win_Rate']:>8.1f}%")
    
    print("="*70)

def plot_portfolio_performance_visuals(aligned_data, key_stats, feature_importance_df=None, 
                                     portfolio_name="Replica", benchmark_name="MTUM"):
    """
    Create the three key visuals: cumulative returns, rolling tracking error, and feature importance
    
    Parameters:
    aligned_data: DataFrame with Portfolio and Benchmark returns
    key_stats: Dictionary with key statistics
    feature_importance_df: DataFrame with feature importance from XGBoost model
    portfolio_name: Name of your portfolio (default: "Replica")
    benchmark_name: Name of benchmark (default: "MTUM")
    """
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + aligned_data['Portfolio']).cumprod()
    benchmark_cumulative = (1 + aligned_data['Benchmark']).cumprod()
    
    # Calculate rolling tracking error (60-day window)
    excess_returns = aligned_data['Portfolio'] - aligned_data['Benchmark']
    rolling_tracking_error = excess_returns.rolling(60).std() * np.sqrt(252)
    
    # Create the plots
    if feature_importance_df is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cumulative Returns Comparison
        axes[0, 0].plot(portfolio_cumulative.index, portfolio_cumulative, 
                       label=f'{portfolio_name}', linewidth=2.5, color='blue')
        axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative, 
                       label=f'{benchmark_name}', linewidth=2.5, color='orange')
        axes[0, 0].set_ylabel('Cumulative Return', fontsize=12)
        axes[0, 0].set_title(f'Cumulative Returns: {portfolio_name} vs {benchmark_name}', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add performance statistics as text
        final_portfolio_return = portfolio_cumulative.iloc[-1] - 1
        final_benchmark_return = benchmark_cumulative.iloc[-1] - 1
        outperformance = final_portfolio_return - final_benchmark_return
        
        stats_text = f'Total Return:\n{portfolio_name}: {final_portfolio_return:.1%}\n{benchmark_name}: {final_benchmark_return:.1%}\nOutperformance: {outperformance:.1%}'
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Rolling Tracking Error
        axes[0, 1].plot(rolling_tracking_error.index, rolling_tracking_error * 100, 
                       color='red', linewidth=2)
        axes[0, 1].axhline(y=key_stats['Tracking_Error'] * 100, color='black', 
                          linestyle='--', alpha=0.7, label=f'Average: {key_stats["Tracking_Error"]:.2%}')
        axes[0, 1].set_ylabel('Tracking Error (%)', fontsize=12)
        axes[0, 1].set_title('Rolling 60-Day Tracking Error', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add tracking error statistics
        te_stats = f'Avg TE: {key_stats["Tracking_Error"]:.2%}\nMax TE: {rolling_tracking_error.max():.2%}\nMin TE: {rolling_tracking_error.min():.2%}'
        axes[0, 1].text(0.02, 0.98, te_stats, transform=axes[0, 1].transAxes, 
                       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 3. Feature Importance Chart
        top_features = feature_importance_df.head(15)  # Top 15 features
        bars = axes[1, 0].barh(range(len(top_features)), top_features['importance'], color='skyblue', edgecolor='navy')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'], fontsize=10)
        axes[1, 0].set_xlabel('Feature Importance', fontsize=12)
        axes[1, 0].set_title('XGBoost Feature Importance (Top 15)', fontsize=14, fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # Add importance values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1, 0].text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # 4. Key Statistics Summary
        axes[1, 1].axis('off')
        
        # Create statistics text
        stats_summary = f"""
        KEY PERFORMANCE METRICS
        
        Information Ratio: {key_stats['Information_Ratio']:.3f}
        Tracking Error: {key_stats['Tracking_Error']:.2%}
        Excess Return: {key_stats['Excess_Return']:.2%}
        
        Portfolio Sharpe: {key_stats['Portfolio_Sharpe']:.3f}
        Benchmark Sharpe: {key_stats['Benchmark_Sharpe']:.3f}
        
        Max Drawdown:
          • Portfolio: {key_stats['Portfolio_Max_Drawdown']:.2%}
          • Benchmark: {key_stats['Benchmark_Max_Drawdown']:.2%}
        
        Win Rate: {key_stats['Win_Rate']:.1f}%
        """
        
        if not np.isnan(key_stats['Average_Holdings']):
            stats_summary += f"\nAverage Holdings: {key_stats['Average_Holdings']:.1f}"
        
        if not np.isnan(key_stats['Annual_Turnover']):
            stats_summary += f"\nAnnual Turnover: {key_stats['Annual_Turnover']:.1%}"
        
        axes[1, 1].text(0.1, 0.9, stats_summary, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
    else:
        # If no feature importance, create 2x1 layout
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Cumulative Returns Comparison
        axes[0].plot(portfolio_cumulative.index, portfolio_cumulative, 
                    label=f'{portfolio_name}', linewidth=2.5, color='blue')
        axes[0].plot(benchmark_cumulative.index, benchmark_cumulative, 
                    label=f'{benchmark_name}', linewidth=2.5, color='orange')
        axes[0].set_ylabel('Cumulative Return', fontsize=12)
        axes[0].set_title(f'Cumulative Returns: {portfolio_name} vs {benchmark_name}', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Rolling Tracking Error
        axes[1].plot(rolling_tracking_error.index, rolling_tracking_error * 100, 
                    color='red', linewidth=2)
        axes[1].axhline(y=key_stats['Tracking_Error'] * 100, color='black', 
                       linestyle='--', alpha=0.7, label=f'Average: {key_stats["Tracking_Error"]:.2%}')
        axes[1].set_ylabel('Tracking Error (%)', fontsize=12)
        axes[1].set_title('Rolling 60-Day Tracking Error', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_portfolio_performance_report(portfolio_returns, benchmark_returns, 
                                        feature_importance_df=None, portfolio_weights_history=None,
                                        portfolio_name="Replica", benchmark_name="MTUM"):
    """
    Generate complete portfolio performance report with key statistics and visuals
    
    Parameters:
    portfolio_returns: Series of portfolio returns
    benchmark_returns: Series of benchmark returns  
    feature_importance_df: DataFrame with XGBoost feature importance
    portfolio_weights_history: DataFrame with historical weights (for turnover calculation)
    portfolio_name: Name of your portfolio
    benchmark_name: Name of benchmark
    """
    
    print(f"\n{'='*70}")
    print(f"PORTFOLIO PERFORMANCE REPORT: {portfolio_name} vs {benchmark_name}")
    print(f"{'='*70}")
    
    # Calculate key statistics
    key_stats, aligned_data = calculate_key_statistics(
        portfolio_returns, benchmark_returns, portfolio_weights_history
    )
    
    # Print statistics table
    print_key_statistics_table(key_stats)
    
    # Create visuals
    print(f"\nGenerating performance visuals...")
    plot_portfolio_performance_visuals(
        aligned_data, key_stats, feature_importance_df, 
        portfolio_name, benchmark_name
    )
    
    return key_stats, aligned_data


###############################################################################################################################################################################
# COMPREHENSIVE BACKTEST ANALYSIS - ENTIRE TEST PERIOD
###############################################################################################################################################################################

def create_full_backtest_analysis(main_backtest_storage, monthly_weights_history, 
                                feature_importance_history, test_period_start, test_period_end):
    """
    Create comprehensive analysis charts for the entire backtest period
    
    Parameters:
    main_backtest_storage: Your backtest storage object
    monthly_weights_history: List of monthly weight dictionaries
    feature_importance_history: List of monthly feature importance DataFrames  
    test_period_start: Start date of test period (first month)
    test_period_end: End date of test period (last month)
    """
    
    print(f"\n📊 Creating Full Backtest Analysis for period: {test_period_start.date()} to {test_period_end.date()}")
    
    # Get all the data
    summary_dfs = main_backtest_storage.get_summary_dataframes()
    
    # 1. Prepare cumulative returns data for entire period
    if 'daily_returns' in summary_dfs and 'daily_benchmark_returns' in summary_dfs:
        portfolio_returns = summary_dfs['daily_returns']
        benchmark_returns = summary_dfs['daily_benchmark_returns']
        print(f"   📈 Portfolio returns: {len(portfolio_returns)} days")
        print(f"   📊 Benchmark returns: {len(benchmark_returns)} days")
        
        # Ensure proper format
        if isinstance(portfolio_returns, pd.DataFrame):
            portfolio_returns = portfolio_returns.iloc[:, 0]  # Take first column
        if isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = benchmark_returns.iloc[:, 0]
            
        # Align the data
        aligned_data = pd.DataFrame({
            'Portfolio': portfolio_returns,
            'Benchmark': benchmark_returns
        }).dropna()
        print(f"   ✅ Successfully aligned data: {len(aligned_data)} days from {aligned_data.index.min().date()} to {aligned_data.index.max().date()}")
        has_benchmark = True
        print(f"   📈 Portfolio data: {len(aligned_data)} days from {aligned_data.index.min().date()} to {aligned_data.index.max().date()}")
        
    else:
        print("   ⚠️  No daily returns data available for full period analysis")
        return None
    
    # 2. Create the comprehensive charts
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Chart 1: Cumulative Returns for Entire Period
    portfolio_cumulative = (1 + aligned_data['Portfolio']).cumprod()
    benchmark_cumulative = (1 + aligned_data['Benchmark']).cumprod()
    
    axes[0, 0].plot(portfolio_cumulative.index, portfolio_cumulative, 
                   label='Portfolio', linewidth=2.5, color='blue')
    axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative, 
                   label='MTUM Benchmark', linewidth=2.5, color='orange')
    
    # Add performance statistics
    final_portfolio_return = portfolio_cumulative.iloc[-1] - 1
    final_benchmark_return = benchmark_cumulative.iloc[-1] - 1
    outperformance = final_portfolio_return - final_benchmark_return
    
    stats_text = f'Total Return:\nPortfolio: {final_portfolio_return:.1%}\nMTUM: {final_benchmark_return:.1%}\nOutperformance: {outperformance:.1%}'
    axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                   fontsize=12, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[0, 0].set_ylabel('Cumulative Return', fontsize=12)
    axes[0, 0].set_title(f'Full Backtest: Cumulative Returns ({test_period_start.date()} to {test_period_end.date()})', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Chart 2: Rolling Tracking Error for Entire Period
    excess_returns = aligned_data['Portfolio'] - aligned_data['Benchmark']
    rolling_tracking_error = excess_returns.rolling(60).std() * np.sqrt(252)  # 60-day rolling, annualized
    
    axes[0, 1].plot(rolling_tracking_error.index, rolling_tracking_error * 100, 
                   color='red', linewidth=2)
    
    avg_tracking_error = rolling_tracking_error.mean()
    axes[0, 1].axhline(y=avg_tracking_error * 100, color='black', 
                      linestyle='--', alpha=0.7, label=f'Average: {avg_tracking_error:.2%}')
    
    axes[0, 1].set_ylabel('Tracking Error (%)', fontsize=12)
    axes[0, 1].set_title('Full Backtest: Rolling 60-Day Tracking Error', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add tracking error statistics
    te_stats = f'Avg: {avg_tracking_error:.2%}\nMax: {rolling_tracking_error.max():.2%}\nMin: {rolling_tracking_error.min():.2%}'
    axes[0, 1].text(0.02, 0.98, te_stats, transform=axes[0, 1].transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Chart 3: Feature Importance Over Time (if available)
    if feature_importance_history and len(feature_importance_history) > 0:
        # Combine all feature importance data
        all_importance = []
        dates = []
        
        for i, fi_data in enumerate(feature_importance_history):
            if fi_data is not None and not fi_data.empty:
                fi_dict = fi_data.set_index('feature')['importance'].to_dict()
                fi_dict['date'] = test_period_start + pd.DateOffset(months=i)
                all_importance.append(fi_dict)
                dates.append(test_period_start + pd.DateOffset(months=i))
        
        if all_importance:
            importance_df = pd.DataFrame(all_importance).set_index('date')
            
            # Plot top 5 most important features over time
            feature_means = importance_df.mean().sort_values(ascending=False)
            top_features = feature_means.head(5).index
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(top_features)))
            
            for i, feature in enumerate(top_features):
                if feature in importance_df.columns:
                    axes[1, 0].plot(importance_df.index, importance_df[feature], 
                                   label=feature, linewidth=2, color=colors[i])
            
            axes[1, 0].set_ylabel('Feature Importance', fontsize=12)
            axes[1, 0].set_title('Feature Importance Evolution Over Time', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Feature Importance Data Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Feature Importance Data Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
    
    # Chart 4: Portfolio Characteristics Over Time
    if monthly_weights_history and len(monthly_weights_history) > 0:
        # Calculate monthly portfolio characteristics
        monthly_chars = []
        
        for i, weights_dict in enumerate(monthly_weights_history):
            if weights_dict:
                month_date = weights_dict.get('Date', test_period_start + pd.DateOffset(months=i))
                weights_only = {k: v for k, v in weights_dict.items() if k not in ['Date', 'month_index']}
                
                num_holdings = sum(1 for w in weights_only.values() if w > 0.001)  # Holdings > 0.1%
                max_weight = max(weights_only.values()) if weights_only else 0
                concentration = sum(sorted(weights_only.values(), reverse=True)[:3])  # Top 3 concentration
                
                monthly_chars.append({
                    'date': pd.to_datetime(month_date),
                    'num_holdings': num_holdings,
                    'max_weight': max_weight,
                    'top3_concentration': concentration
                })
        
        if monthly_chars:
            chars_df = pd.DataFrame(monthly_chars).set_index('date')
            
            # Plot number of holdings over time
            ax4_twin = axes[1, 1].twinx()
            
            line1 = axes[1, 1].plot(chars_df.index, chars_df['num_holdings'], 
                                   'b-', linewidth=2, label='Number of Holdings')
            line2 = ax4_twin.plot(chars_df.index, chars_df['max_weight'] * 100, 
                                 'r-', linewidth=2, label='Max Weight (%)')
            
            axes[1, 1].set_ylabel('Number of Holdings', fontsize=12, color='blue')
            ax4_twin.set_ylabel('Max Weight (%)', fontsize=12, color='red')
            axes[1, 1].set_title('Portfolio Characteristics Over Time', fontsize=14, fontweight='bold')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[1, 1].legend(lines, labels, loc='upper left')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Portfolio Characteristics Data Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Portfolio Characteristics Data Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
    
    plt.tight_layout()
    
    # Save the comprehensive chart
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    chart_filename = f'full_backtest_analysis_{timestamp}.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Saved comprehensive analysis chart: {chart_filename}")
    
    plt.show()
    
    return chart_filename

# Usage: Add this at the end of your backtest, after the main loop completes    

# Example usage:
"""
# Generate complete performance report
key_statistics, performance_data = generate_portfolio_performance_report(
    portfolio_returns=portfolio_historical_returns,
    benchmark_returns=benchmark_returns,  # Your MTUM benchmark
    feature_importance_df=feature_importance,  # From XGBoost model
    portfolio_weights_history=None,  # Optional: DataFrame with historical weights
    portfolio_name="My Replica Portfolio",
    benchmark_name="MTUM"
)

# The report will show:
# 1. Formatted table with all key statistics
# 2. Cumulative returns chart (Portfolio vs MTUM)
# 3. Rolling tracking error over time
# 4. XGBoost feature importance chart
# 5. Summary statistics panel
"""