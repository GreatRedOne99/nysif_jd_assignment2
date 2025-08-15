import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



def prepare_cvxpy_data(df, selected_securities, date_col='Date', security_col='Ticker',  return_col='return', lookback_days=252):
    """
    Prepare returns matrix and statistics for CVXPY optimization
    
    Parameters:
    df: DataFrame with daily returns data
    selected_securities: list of securities to include in portfolio
    lookback_days: number of days to look back (252 = 1 year of trading days)
    """
    
    # Filter for selected securities only
    portfolio_data = df[df[security_col].isin(selected_securities)].copy()
    portfolio_data[date_col] = pd.to_datetime(portfolio_data[date_col])
    
    # Get the last year of data
    max_date = portfolio_data[date_col].max()
    start_date = max_date - timedelta(days=int(lookback_days * 1.4))  # Buffer for weekends/holidays
    
    recent_data = portfolio_data[portfolio_data[date_col] >= start_date].copy()
    
    # Create returns matrix (dates x securities)
    returns_matrix = recent_data.pivot(index=date_col, columns=security_col, values=return_col)
    
    # Take last lookback_days of data
    returns_matrix = returns_matrix.tail(lookback_days)
    
    # Remove securities with too much missing data (>20% missing)
    missing_threshold = 0.2
    valid_securities = returns_matrix.columns[returns_matrix.isnull().mean() < missing_threshold].tolist()
    returns_matrix = returns_matrix[valid_securities]
    
    # Forward fill small gaps, then drop remaining NaN rows
    returns_matrix = returns_matrix.fillna(method='ffill').dropna()
    
    print(f"Returns matrix shape: {returns_matrix.shape}")
    print(f"Date range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
    print(f"Securities included: {len(returns_matrix.columns)}")
    
    # Calculate statistics for optimization
    mean_returns = returns_matrix.mean().values * 252  # Annualized expected returns
    cov_matrix = returns_matrix.cov().values * 252     # Annualized covariance matrix
    securities = returns_matrix.columns.tolist()
    
    return returns_matrix, mean_returns, cov_matrix, securities

def optimize_portfolio_cvxpy(mean_returns, cov_matrix, securities, risk_aversion=1.0, l1_penalty=0.01,  max_weight=0.3, min_weight=0.001, target_return=None, max_positions=None):
    """
    Mean-variance optimization with L1 regularization using CVXPY
    
    Parameters:
    mean_returns: array of expected returns
    cov_matrix: covariance matrix
    securities: list of security names
    risk_aversion: risk aversion parameter (higher = more conservative)
    l1_penalty: L1 regularization strength (higher = more sparse, fewer positions)
    max_weight: maximum weight per security
    min_weight: minimum weight threshold (positions below this become 0)
    target_return: target portfolio return (optional)
    max_positions: maximum number of positions (optional)
    """
    
    n = len(securities)
    
    # Decision variable: portfolio weights
    w = cp.Variable(n, nonneg=True)  # Non-negative weights (no shorting)
    
    # Portfolio return and risk
    portfolio_return = mean_returns.T @ w
    portfolio_risk = cp.quad_form(w, cov_matrix)
    
    # Objective: Maximize utility with L1 penalty for sparsity
    # Utility = Expected Return - (Risk Aversion/2) * Variance - L1 Penalty * ||w||_1
    objective = cp.Maximize(portfolio_return - (risk_aversion/2) * portfolio_risk - l1_penalty * cp.norm(w, 1))
    
    # Constraints
    constraints = [
        cp.sum(w) == 1.0,           # Weights sum to 1
        w >= 0,                     # No shorting (already specified in variable definition)
        w <= max_weight             # Maximum weight constraint
    ]
    
    # Optional: Target return constraint
    if target_return is not None:
        constraints.append(portfolio_return >= target_return)
    
    # Optional: Maximum number of positions (cardinality constraint approximation)
    if max_positions is not None and max_positions < n:
        # This is a heuristic - we'll solve and then keep only top positions
        pass
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    
    # Solve with different solvers if needed
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except:
            problem.solve(solver=cp.OSQP, verbose=False)
    
    if problem.status not in ["infeasible", "unbounded"]:
        optimal_weights = w.value
        
        # Apply minimum weight threshold (sparsity post-processing)
        optimal_weights[optimal_weights < min_weight] = 0
        
        # Renormalize to ensure weights sum to 1
        if np.sum(optimal_weights) > 0:
            optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Apply max positions constraint if specified
        if max_positions is not None and max_positions < n:
            # Keep only the largest weights
            top_indices = np.argsort(optimal_weights)[-max_positions:]
            sparse_weights = np.zeros(n)
            sparse_weights[top_indices] = optimal_weights[top_indices]
            # Renormalize
            sparse_weights = sparse_weights / np.sum(sparse_weights)
            optimal_weights = sparse_weights
        
        # Calculate portfolio metrics
        portfolio_ret = np.sum(optimal_weights * mean_returns)
        portfolio_var = np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
        portfolio_vol = np.sqrt(portfolio_var)
        sharpe_ratio = portfolio_ret / portfolio_vol if portfolio_vol > 0 else 0
        
        # Create results dataframe
        weights_df = pd.DataFrame({
            'Security': securities,
            'Weight': optimal_weights,
            'Expected_Return': mean_returns,
            'Volatility': np.sqrt(np.diag(cov_matrix))
        })
        
        # Filter out zero weights and sort
        weights_df = weights_df[weights_df['Weight'] > 0].sort_values('Weight', ascending=False).reset_index(drop=True)
        
        print(f"\nCVXPY Optimization Results:")
        print(f"Solver Status: {problem.status}")
        print(f"Expected Annual Return: {portfolio_ret:.2%}")
        print(f"Expected Annual Volatility: {portfolio_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Number of positions: {len(weights_df)}")
        print(f"Largest position: {weights_df.iloc[0]['Weight']:.2%}")
        print(f"Portfolio concentration (top 5): {weights_df.head(5)['Weight'].sum():.2%}")
        
        return weights_df, portfolio_ret, portfolio_vol, sharpe_ratio, problem.status
    
    else:
        print(f"Optimization failed: {problem.status}")
        return None, None, None, None, problem.status

def compare_regularization_levels(mean_returns, cov_matrix, securities, l1_penalties=[0.0, 0.005, 0.01, 0.02, 0.05],  risk_aversion=1.0, max_weight=0.3):
    """
    Compare portfolio optimization results across different L1 penalty levels
    """
    
    results = []
    
    for l1_penalty in l1_penalties:
        weights_df, ret, vol, sharpe, status = optimize_portfolio_cvxpy(
            mean_returns, cov_matrix, securities,
            risk_aversion=risk_aversion,
            l1_penalty=l1_penalty,
            max_weight=max_weight
        )
        
        if weights_df is not None:
            results.append({
                'L1_Penalty': l1_penalty,
                'Return': ret,
                'Volatility': vol,
                'Sharpe': sharpe,
                'Num_Positions': len(weights_df),
                'Max_Weight': weights_df['Weight'].max(),
                'Top5_Concentration': weights_df.head(5)['Weight'].sum(),
                'Status': status
            })
    
    comparison_df = pd.DataFrame(results)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Number of positions vs L1 penalty
    axes[0, 0].plot(comparison_df['L1_Penalty'], comparison_df['Num_Positions'], 'bo-')
    axes[0, 0].set_xlabel('L1 Penalty')
    axes[0, 0].set_ylabel('Number of Positions')
    axes[0, 0].set_title('Sparsity Effect')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sharpe ratio vs L1 penalty
    axes[0, 1].plot(comparison_df['L1_Penalty'], comparison_df['Sharpe'], 'ro-')
    axes[0, 1].set_xlabel('L1 Penalty')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].set_title('Performance vs Sparsity Trade-off')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Risk-Return scatter
    axes[1, 0].scatter(comparison_df['Volatility'], comparison_df['Return'], 
                       c=comparison_df['L1_Penalty'], cmap='viridis', s=100)
    axes[1, 0].set_xlabel('Volatility')
    axes[1, 0].set_ylabel('Expected Return')
    axes[1, 0].set_title('Risk-Return by L1 Penalty')
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('L1 Penalty')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Concentration vs L1 penalty
    axes[1, 1].plot(comparison_df['L1_Penalty'], comparison_df['Top5_Concentration'], 'go-')
    axes[1, 1].set_xlabel('L1 Penalty')
    axes[1, 1].set_ylabel('Top 5 Concentration')
    axes[1, 1].set_title('Portfolio Concentration')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nRegularization Comparison:")
    print(comparison_df.round(4))
    
    return comparison_df

def plot_cvxpy_portfolio(weights_df, portfolio_return, portfolio_vol, sharpe_ratio):
    """
    Plot portfolio analysis for CVXPY optimization results
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio weights pie chart
    top_positions = weights_df.head(10)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_positions)))
    axes[0, 0].pie(top_positions['Weight'], labels=top_positions['Security'], 
                   autopct='%1.1f%%', colors=colors)
    axes[0, 0].set_title('Portfolio Weights (Top 10 Positions)')
    
    # Portfolio weights bar chart
    top_15 = weights_df.head(15)
    bars = axes[0, 1].barh(range(len(top_15)), top_15['Weight'], color='skyblue')
    axes[0, 1].set_yticks(range(len(top_15)))
    axes[0, 1].set_yticklabels(top_15['Security'])
    axes[0, 1].set_xlabel('Weight')
    axes[0, 1].set_title('Portfolio Weights (Top 15)')
    axes[0, 1].invert_yaxis()
    
    # Add weight labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[0, 1].text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1%}', ha='left', va='center', fontsize=9)
    
    # Risk-Return scatter of individual securities
    axes[1, 0].scatter(weights_df['Volatility'], weights_df['Expected_Return'], 
                       s=weights_df['Weight']*1000, alpha=0.6, c='skyblue', edgecolor='navy')
    axes[1, 0].scatter(portfolio_vol, portfolio_return, color='red', s=200, 
                       marker='*', label='Portfolio', edgecolor='black')
    axes[1, 0].set_xlabel('Annual Volatility')
    axes[1, 0].set_ylabel('Annual Expected Return')
    axes[1, 0].set_title('Risk-Return (Bubble Size = Weight)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Portfolio statistics
    axes[1, 1].axis('off')
    n_positions = len(weights_df)
    max_weight = weights_df['Weight'].max()
    top5_concentration = weights_df.head(5)['Weight'].sum()
    diversification_ratio = 1 / np.sum(weights_df['Weight']**2)
    
    stats_text = f"""
    CVXPY Portfolio Statistics:
    
    Expected Annual Return: {portfolio_return:.2%}
    Expected Annual Volatility: {portfolio_vol:.2%}
    Sharpe Ratio: {sharpe_ratio:.3f}
    
    Number of Positions: {n_positions}
    Largest Position: {max_weight:.2%}
    Top 5 Concentration: {top5_concentration:.2%}
    
    Diversification Ratio: {diversification_ratio:.1f}
    
    Portfolio Characteristics:
    - L1 Regularization creates sparse portfolio
    - No short positions (long-only)
    - Position size constraints applied
    - Mean-variance optimized
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def calculate_portfolio_returns(optimal_weights, returns_matrix):
    """
    Calculate historical portfolio returns based on weights
    
    Parameters:
    optimal_weights: DataFrame with Security and Weight columns
    returns_matrix: DataFrame with historical returns (dates x securities)
    """
    
    # Align securities and create weight vector
    portfolio_securities = optimal_weights['Security'].tolist()
    portfolio_weights = optimal_weights.set_index('Security')['Weight']
    
    # Filter returns matrix for portfolio securities
    aligned_returns = returns_matrix[portfolio_securities]
    
    # Calculate portfolio returns
    portfolio_returns = (aligned_returns * portfolio_weights).sum(axis=1)
    
    return portfolio_returns

def calculate_benchmark_comparison_metrics(portfolio_returns, benchmark_returns, risk_free_rate=0.02):
    """
    Calculate comprehensive comparison metrics between portfolio and benchmark
    
    Parameters:
    portfolio_returns: Series of portfolio returns
    benchmark_returns: Series, DataFrame, or array-like benchmark data
    risk_free_rate: Annual risk-free rate (default 2%)
    """
    
    # Handle different benchmark_returns formats
    if not isinstance(benchmark_returns, pd.Series):
        print("Converting benchmark_returns to proper Series format...")
        if isinstance(benchmark_returns, pd.DataFrame):
            # Try to auto-detect format
            if 'return' in benchmark_returns.columns:
                if 'Date' in benchmark_returns.columns:
                    benchmark_returns = prepare_benchmark_returns(benchmark_returns, 'Date', 'return')
                else:
                    # Assume first column is date, 'return' column is returns
                    date_col = benchmark_returns.columns[0]
                    benchmark_returns = prepare_benchmark_returns(benchmark_returns, date_col, 'return')
            else:
                raise ValueError("Cannot auto-detect benchmark return format. Please use prepare_benchmark_returns() first.")
        else:
            # Array-like - convert to Series with portfolio index
            benchmark_returns = pd.Series(benchmark_returns, index=portfolio_returns.index)
    
    # Align the series by date
    aligned_data = pd.DataFrame({
        'Portfolio': portfolio_returns,
        'Benchmark': benchmark_returns
    }).dropna()
    
    if aligned_data.empty:
        raise ValueError("No overlapping dates between portfolio and benchmark returns")
    
    print(f"Aligned data shape: {aligned_data.shape}")
    print(f"Date range: {aligned_data.index.min()} to {aligned_data.index.max()}")
    
    portfolio_ret = aligned_data['Portfolio']
    benchmark_ret = aligned_data['Benchmark']
    
    # Annualization factor
    periods_per_year = 252  # Trading days
    
    # Basic statistics
    portfolio_annual_return = portfolio_ret.mean() * periods_per_year
    benchmark_annual_return = benchmark_ret.mean() * periods_per_year
    
    portfolio_annual_vol = portfolio_ret.std() * np.sqrt(periods_per_year)
    benchmark_annual_vol = benchmark_ret.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratios
    portfolio_sharpe = (portfolio_annual_return - risk_free_rate) / portfolio_annual_vol
    benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_annual_vol
    
    # Excess returns (portfolio vs benchmark)
    excess_returns = portfolio_ret - benchmark_ret
    excess_annual_return = excess_returns.mean() * periods_per_year
    excess_annual_vol = excess_returns.std() * np.sqrt(periods_per_year)
    information_ratio = excess_annual_return / excess_annual_vol if excess_annual_vol > 0 else 0
    
    # Tracking error
    tracking_error = excess_annual_vol
    
    # Beta and Alpha (CAPM)
    beta, alpha_daily, r_value, p_value, std_err = stats.linregress(benchmark_ret, portfolio_ret)
    alpha_annual = alpha_daily * periods_per_year
    r_squared = r_value ** 2
    
    # Downside metrics
    downside_threshold = 0  # Daily threshold for downside deviation
    portfolio_downside_returns = portfolio_ret[portfolio_ret < downside_threshold]
    benchmark_downside_returns = benchmark_ret[benchmark_ret < downside_threshold]
    
    portfolio_downside_vol = np.sqrt(np.mean(portfolio_downside_returns**2)) * np.sqrt(periods_per_year)
    benchmark_downside_vol = np.sqrt(np.mean(benchmark_downside_returns**2)) * np.sqrt(periods_per_year)
    
    # Sortino ratios
    portfolio_sortino = (portfolio_annual_return - risk_free_rate) / portfolio_downside_vol if portfolio_downside_vol > 0 else np.inf
    benchmark_sortino = (benchmark_annual_return - risk_free_rate) / benchmark_downside_vol if benchmark_downside_vol > 0 else np.inf
    
    # Maximum drawdown
    def calculate_max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    portfolio_max_drawdown = calculate_max_drawdown(portfolio_ret)
    benchmark_max_drawdown = calculate_max_drawdown(benchmark_ret)
    
    # Calmar ratio (Annual return / Max Drawdown)
    portfolio_calmar = portfolio_annual_return / abs(portfolio_max_drawdown) if portfolio_max_drawdown != 0 else np.inf
    benchmark_calmar = benchmark_annual_return / abs(benchmark_max_drawdown) if benchmark_max_drawdown != 0 else np.inf
    
    # Up/Down capture ratios
    up_markets = benchmark_ret > 0
    down_markets = benchmark_ret < 0
    
    if up_markets.sum() > 0:
        up_capture = (portfolio_ret[up_markets].mean() / benchmark_ret[up_markets].mean()) * 100
    else:
        up_capture = np.nan
        
    if down_markets.sum() > 0:
        down_capture = (portfolio_ret[down_markets].mean() / benchmark_ret[down_markets].mean()) * 100
    else:
        down_capture = np.nan
    
    # Win rate
    win_rate = (excess_returns > 0).mean() * 100
    
    # Compile all metrics
    metrics = {
        'Portfolio_Annual_Return': portfolio_annual_return,
        'Benchmark_Annual_Return': benchmark_annual_return,
        'Excess_Annual_Return': excess_annual_return,
        
        'Portfolio_Annual_Volatility': portfolio_annual_vol,
        'Benchmark_Annual_Volatility': benchmark_annual_vol,
        'Tracking_Error': tracking_error,
        
        'Portfolio_Sharpe_Ratio': portfolio_sharpe,
        'Benchmark_Sharpe_Ratio': benchmark_sharpe,
        'Information_Ratio': information_ratio,
        
        'Portfolio_Sortino_Ratio': portfolio_sortino,
        'Benchmark_Sortino_Ratio': benchmark_sortino,
        
        'Beta': beta,
        'Alpha_Annual': alpha_annual,
        'R_Squared': r_squared,
        
        'Portfolio_Max_Drawdown': portfolio_max_drawdown,
        'Benchmark_Max_Drawdown': benchmark_max_drawdown,
        
        'Portfolio_Calmar_Ratio': portfolio_calmar,
        'Benchmark_Calmar_Ratio': benchmark_calmar,
        
        'Up_Capture_Ratio': up_capture,
        'Down_Capture_Ratio': down_capture,
        'Win_Rate': win_rate,
        
        'Portfolio_Downside_Volatility': portfolio_downside_vol,
        'Benchmark_Downside_Volatility': benchmark_downside_vol
    }
    
    return metrics, aligned_data

def print_benchmark_comparison(metrics):
    """
    Print formatted benchmark comparison table
    """
    
    print("\n" + "="*80)
    print("PORTFOLIO vs BENCHMARK COMPARISON")
    print("="*80)
    
    print(f"{'Metric':<30} {'Portfolio':<15} {'Benchmark':<15} {'Difference':<15}")
    print("-"*80)
    
    # Returns
    print(f"{'Annual Return':<30} {metrics['Portfolio_Annual_Return']:<15.2%} {metrics['Benchmark_Annual_Return']:<15.2%} {metrics['Excess_Annual_Return']:<15.2%}")
    
    # Risk
    print(f"{'Annual Volatility':<30} {metrics['Portfolio_Annual_Volatility']:<15.2%} {metrics['Benchmark_Annual_Volatility']:<15.2%} {metrics['Portfolio_Annual_Volatility']-metrics['Benchmark_Annual_Volatility']:<15.2%}")
    
    # Risk-adjusted metrics
    print(f"{'Sharpe Ratio':<30} {metrics['Portfolio_Sharpe_Ratio']:<15.3f} {metrics['Benchmark_Sharpe_Ratio']:<15.3f} {metrics['Portfolio_Sharpe_Ratio']-metrics['Benchmark_Sharpe_Ratio']:<15.3f}")
    print(f"{'Sortino Ratio':<30} {metrics['Portfolio_Sortino_Ratio']:<15.3f} {metrics['Benchmark_Sortino_Ratio']:<15.3f} {metrics['Portfolio_Sortino_Ratio']-metrics['Benchmark_Sortino_Ratio']:<15.3f}")
    
    # CAPM metrics
    print(f"{'Beta':<30} {metrics['Beta']:<15.3f} {'1.000':<15} {metrics['Beta']-1:<15.3f}")
    print(f"{'Alpha (Annual)':<30} {metrics['Alpha_Annual']:<15.2%} {'0.00%':<15} {metrics['Alpha_Annual']:<15.2%}")
    print(f"{'R-Squared':<30} {metrics['R_Squared']:<15.3f} {'-':<15} {'-':<15}")
    
    # Drawdown metrics
    print(f"{'Max Drawdown':<30} {metrics['Portfolio_Max_Drawdown']:<15.2%} {metrics['Benchmark_Max_Drawdown']:<15.2%} {metrics['Portfolio_Max_Drawdown']-metrics['Benchmark_Max_Drawdown']:<15.2%}")
    print(f"{'Calmar Ratio':<30} {metrics['Portfolio_Calmar_Ratio']:<15.3f} {metrics['Benchmark_Calmar_Ratio']:<15.3f} {metrics['Portfolio_Calmar_Ratio']-metrics['Benchmark_Calmar_Ratio']:<15.3f}")
    
    # Additional metrics
    print(f"{'Tracking Error':<30} {metrics['Tracking_Error']:<15.2%} {'-':<15} {'-':<15}")
    print(f"{'Information Ratio':<30} {metrics['Information_Ratio']:<15.3f} {'-':<15} {'-':<15}")
    print(f"{'Up Capture Ratio':<30} {metrics['Up_Capture_Ratio']:<15.1f}% {'-':<15} {'-':<15}")
    print(f"{'Down Capture Ratio':<30} {metrics['Down_Capture_Ratio']:<15.1f}% {'-':<15} {'-':<15}")
    print(f"{'Win Rate':<30} {metrics['Win_Rate']:<15.1f}% {'-':<15} {'-':<15}")
    
    print("="*80)

def plot_benchmark_comparison(aligned_data, weights_df, portfolio_vol, portfolio_return, sharpe_ratio, metrics, portfolio_name="Optimized Portfolio"):
    """
    Create comprehensive benchmark comparison plots
    """
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + aligned_data['Portfolio']).cumprod()
    benchmark_cumulative = (1 + aligned_data['Benchmark']).cumprod()
    
    # Calculate rolling metrics
    window = 60  # 60-day rolling window
    rolling_excess = aligned_data['Portfolio'] - aligned_data['Benchmark']
    rolling_excess_mean = rolling_excess.rolling(window).mean() * 252
    rolling_tracking_error = rolling_excess.rolling(window).std() * np.sqrt(252)
    rolling_info_ratio = rolling_excess_mean / rolling_tracking_error
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Cumulative returns comparison
    axes[0, 0].plot(portfolio_cumulative.index, portfolio_cumulative, label=portfolio_name, linewidth=2)
    axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative, label='Benchmark', linewidth=2)
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].set_title('Cumulative Returns Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Excess returns over time
    axes[0, 1].plot(rolling_excess_mean.index, rolling_excess_mean * 100, color='green', linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('Excess Return (%)')
    axes[0, 1].set_title('Rolling 60-Day Excess Return (Annualized)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rolling information ratio
    axes[1, 0].plot(rolling_info_ratio.index, rolling_info_ratio, color='orange', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Good IR > 0.5')
    axes[1, 0].set_ylabel('Information Ratio')
    axes[1, 0].set_title('Rolling 60-Day Information Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot: Portfolio vs Benchmark returns
    axes[1, 1].scatter(aligned_data['Benchmark'], aligned_data['Portfolio'], alpha=0.6, s=20)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(aligned_data['Benchmark'], aligned_data['Portfolio'])
    line_x = np.array([aligned_data['Benchmark'].min(), aligned_data['Benchmark'].max()])
    line_y = slope * line_x + intercept
    axes[1, 1].plot(line_x, line_y, 'r-', linewidth=2, label=f'Beta = {slope:.3f}')
    axes[1, 1].plot([-0.1, 0.1], [-0.1, 0.1], 'k--', alpha=0.5, label='Perfect Correlation')
    
    axes[1, 1].set_xlabel('Benchmark Daily Return')
    axes[1, 1].set_ylabel('Portfolio Daily Return')
    axes[1, 1].set_title('Portfolio vs Benchmark Scatter (Beta Analysis)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Performance metrics bar chart
    metrics_to_plot = ['Portfolio_Sharpe_Ratio', 'Benchmark_Sharpe_Ratio', 
                       'Portfolio_Calmar_Ratio', 'Benchmark_Calmar_Ratio']
    metric_values = [metrics[m] for m in metrics_to_plot]
    metric_labels = ['Portfolio\nSharpe', 'Benchmark\nSharpe', 'Portfolio\nCalmar', 'Benchmark\nCalmar']
    
    bars = axes[2, 0].bar(range(len(metric_labels)), metric_values, 
                          color=['blue', 'orange', 'blue', 'orange'], alpha=0.7)
    axes[2, 0].set_xticks(range(len(metric_labels)))
    axes[2, 0].set_xticklabels(metric_labels)
    axes[2, 0].set_ylabel('Ratio Value')
    axes[2, 0].set_title('Risk-Adjusted Performance Comparison')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[2, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
    
    # Drawdown comparison
    def calculate_drawdown_series(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    portfolio_dd = calculate_drawdown_series(aligned_data['Portfolio'])
    benchmark_dd = calculate_drawdown_series(aligned_data['Benchmark'])
    
    axes[2, 1].fill_between(portfolio_dd.index, portfolio_dd, 0, alpha=0.3, label=portfolio_name, color='blue')
    axes[2, 1].fill_between(benchmark_dd.index, benchmark_dd, 0, alpha=0.3, label='Benchmark', color='orange')
    axes[2, 1].set_ylabel('Drawdown')
    axes[2, 1].set_title('Drawdown Comparison')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    """
    Plot portfolio analysis for CVXPY optimization results
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio weights pie chart
    top_positions = weights_df.head(10)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_positions)))
    axes[0, 0].pie(top_positions['Weight'], labels=top_positions['Security'], 
                   autopct='%1.1f%%', colors=colors)
    axes[0, 0].set_title('Portfolio Weights (Top 10 Positions)')
    
    # Portfolio weights bar chart
    top_15 = weights_df.head(15)
    bars = axes[0, 1].barh(range(len(top_15)), top_15['Weight'], color='skyblue')
    axes[0, 1].set_yticks(range(len(top_15)))
    axes[0, 1].set_yticklabels(top_15['Security'])
    axes[0, 1].set_xlabel('Weight')
    axes[0, 1].set_title('Portfolio Weights (Top 15)')
    axes[0, 1].invert_yaxis()
    
    # Add weight labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[0, 1].text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1%}', ha='left', va='center', fontsize=9)
    
    # Risk-Return scatter of individual securities
    axes[1, 0].scatter(weights_df['Volatility'], weights_df['Expected_Return'], 
                       s=weights_df['Weight']*1000, alpha=0.6, c='skyblue', edgecolor='navy')
    axes[1, 0].scatter(portfolio_vol, portfolio_return, color='red', s=200, 
                       marker='*', label='Portfolio', edgecolor='black')
    axes[1, 0].set_xlabel('Annual Volatility')
    axes[1, 0].set_ylabel('Annual Expected Return')
    axes[1, 0].set_title('Risk-Return (Bubble Size = Weight)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Portfolio statistics
    axes[1, 1].axis('off')
    n_positions = len(weights_df)
    max_weight = weights_df['Weight'].max()
    top5_concentration = weights_df.head(5)['Weight'].sum()
    diversification_ratio = 1 / np.sum(weights_df['Weight']**2)
    
    stats_text = f"""
    CVXPY Portfolio Statistics:
    
    Expected Annual Return: {portfolio_return:.2%}
    Expected Annual Volatility: {portfolio_vol:.2%}
    Sharpe Ratio: {sharpe_ratio:.3f}
    
    Number of Positions: {n_positions}
    Largest Position: {max_weight:.2%}
    Top 5 Concentration: {top5_concentration:.2%}
    
    Diversification Ratio: {diversification_ratio:.1f}
    
    Portfolio Characteristics:
    - L1 Regularization creates sparse portfolio
    - No short positions (long-only)
    - Position size constraints applied
    - Mean-variance optimized
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()    
def prepare_benchmark_returns(benchmark_data, date_col='Date', return_col='return',  benchmark_ticker=None):
    """
    Prepare benchmark returns in the correct format for comparison
    
    Parameters:
    benchmark_data: DataFrame, Series, or array-like with benchmark data
    date_col: column name for dates (if DataFrame)
    return_col: column name for returns (if DataFrame) 
    benchmark_ticker: specific ticker to extract if multiple securities in DataFrame
    
    Returns:
    pandas Series with datetime index and benchmark returns
    """
    
    if isinstance(benchmark_data, pd.Series):
        # Already a Series - just ensure datetime index
        if not isinstance(benchmark_data.index, pd.DatetimeIndex):
            print("Warning: Converting Series index to datetime")
            benchmark_data.index = pd.to_datetime(benchmark_data.index)
        return benchmark_data
    
    elif isinstance(benchmark_data, pd.DataFrame):
        # DataFrame case - need to extract benchmark returns
        benchmark_df = benchmark_data.copy()
        
        # Convert date column to datetime if needed
        if date_col in benchmark_df.columns:
            benchmark_df[date_col] = pd.to_datetime(benchmark_df[date_col])
        
        # Case 1: Single benchmark in DataFrame
        if benchmark_ticker is None:
            if date_col in benchmark_df.columns and return_col in benchmark_df.columns:
                # Standard case: Date and return columns
                benchmark_series = benchmark_df.set_index(date_col)[return_col]
            else:
                # Try to find return-like column
                return_cols = [col for col in benchmark_df.columns if 'return' in col.lower()]
                if return_cols:
                    return_col = return_cols[0]
                    print(f"Using {return_col} as benchmark return column")
                    benchmark_series = benchmark_df.set_index(date_col)[return_col]
                else:
                    raise ValueError("Could not identify return column in benchmark DataFrame")
        
        # Case 2: Multiple securities, need to filter for benchmark
        else:
            if 'Ticker' in benchmark_df.columns or 'ticker' in benchmark_df.columns:
                ticker_col = 'Ticker' if 'Ticker' in benchmark_df.columns else 'ticker'
                benchmark_subset = benchmark_df[benchmark_df[ticker_col] == benchmark_ticker]
                if benchmark_subset.empty:
                    raise ValueError(f"Benchmark ticker {benchmark_ticker} not found in data")
                benchmark_series = benchmark_subset.set_index(date_col)[return_col]
            else:
                raise ValueError("Need ticker column to identify specific benchmark")
        
        return benchmark_series
    
    else:
        # Array-like case - assume it aligns with portfolio returns dates
        print("Warning: benchmark_data is array-like. Assuming it aligns with portfolio dates.")
        return pd.Series(benchmark_data)    