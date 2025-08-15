import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_momentum_factors(returns_matrix, lookback_periods=[21, 63, 126, 252]):
    """
    Calculate momentum factors for expected returns
    
    Parameters:
    returns_matrix: DataFrame with returns (dates x securities)
    lookback_periods: list of lookback periods for momentum calculation
    """
    
    momentum_factors = pd.DataFrame(index=returns_matrix.columns)
    
    for period in lookback_periods:
        if len(returns_matrix) >= period:
            # Calculate cumulative returns over the period
            cum_returns = (1 + returns_matrix.tail(period)).prod() - 1
            momentum_factors[f'momentum_{period}d'] = cum_returns
        else:
            momentum_factors[f'momentum_{period}d'] = 0
    
    return momentum_factors

def calculate_mean_reversion_factors(returns_matrix, lookback_periods=[21, 63]):
    """
    Calculate mean reversion factors (negative momentum for short-term)
    """
    
    reversion_factors = pd.DataFrame(index=returns_matrix.columns)
    
    for period in lookback_periods:
        if len(returns_matrix) >= period:
            # Short-term reversal: negative of recent performance
            recent_returns = (1 + returns_matrix.tail(period)).prod() - 1
            reversion_factors[f'reversion_{period}d'] = -recent_returns
        else:
            reversion_factors[f'reversion_{period}d'] = 0
    
    return reversion_factors

def calculate_volatility_factors(returns_matrix, lookback_periods=[21, 63, 126]):
    """
    Calculate volatility-based factors
    """
    
    vol_factors = pd.DataFrame(index=returns_matrix.columns)
    
    for period in lookback_periods:
        if len(returns_matrix) >= period:
            vol = returns_matrix.tail(period).std() * np.sqrt(252)
            vol_factors[f'volatility_{period}d'] = vol
            
            # Risk-adjusted momentum (return/vol) - handle division by zero
            cum_ret = (1 + returns_matrix.tail(period)).prod() - 1
            
            # Safe division - avoid the ambiguous Series boolean error
            risk_adj_momentum = pd.Series(index=vol.index, dtype=float)
            for ticker in vol.index:
                if vol[ticker] > 0:
                    risk_adj_momentum[ticker] = cum_ret[ticker] / vol[ticker]
                else:
                    risk_adj_momentum[ticker] = 0
            
            vol_factors[f'risk_adj_momentum_{period}d'] = risk_adj_momentum
        else:
            vol_factors[f'volatility_{period}d'] = 0
            vol_factors[f'risk_adj_momentum_{period}d'] = 0
    
    return vol_factors

def build_factor_model_returns(returns_matrix, method='multi_factor', 
                              shrinkage_factor=0.3, risk_aversion=3.0):
    """
    Build ex-ante expected returns using factor models
    
    Parameters:
    returns_matrix: DataFrame with historical returns
    method: 'historical', 'shrinkage', 'momentum', 'mean_reversion', 'multi_factor', 'black_litterman'
    shrinkage_factor: shrinkage towards equal-weighted returns (0-1)
    risk_aversion: risk aversion parameter for Black-Litterman
    """
    
    securities = returns_matrix.columns
    historical_means = returns_matrix.mean() * 252  # Annualized historical means
    
    if method == 'historical':
        # Simple historical means (what we were using before)
        expected_returns = historical_means
        
    elif method == 'shrinkage':
        # James-Stein shrinkage estimator
        grand_mean = historical_means.mean()
        expected_returns = (1 - shrinkage_factor) * historical_means + shrinkage_factor * grand_mean
        
    elif method == 'momentum':
        # Momentum-based expected returns
        momentum_factors = calculate_momentum_factors(returns_matrix)
        
        # Combine different momentum signals
        momentum_score = (
            0.4 * momentum_factors['momentum_21d'] +
            0.3 * momentum_factors['momentum_63d'] + 
            0.2 * momentum_factors['momentum_126d'] +
            0.1 * momentum_factors['momentum_252d']
        )
        
        # Scale momentum to reasonable expected return range
        momentum_scaled = momentum_score * 0.5  # Scale factor
        expected_returns = historical_means * 0.3 + momentum_scaled * 0.7
        
    elif method == 'mean_reversion':
        # Mean reversion model
        reversion_factors = calculate_mean_reversion_factors(returns_matrix)
        
        reversion_score = (
            0.6 * reversion_factors['reversion_21d'] +
            0.4 * reversion_factors['reversion_63d']
        )
        
        expected_returns = historical_means + reversion_score * 0.3
        
    elif method == 'multi_factor':
        # Multi-factor model combining momentum, mean reversion, and volatility
        momentum_factors = calculate_momentum_factors(returns_matrix)
        reversion_factors = calculate_mean_reversion_factors(returns_matrix)
        vol_factors = calculate_volatility_factors(returns_matrix)
        
        # Create factor score
        factor_score = (
            0.3 * momentum_factors['momentum_126d'] +  # Medium-term momentum
            0.2 * reversion_factors['reversion_21d'] +  # Short-term reversion
            0.2 * vol_factors['risk_adj_momentum_63d'] + # Risk-adjusted momentum
            -0.1 * vol_factors['volatility_63d'] +      # Low vol premium
            0.2 * momentum_factors['momentum_252d']     # Long-term trend
        )
        
        # Combine with shrunk historical means
        shrunk_historical = (1 - shrinkage_factor) * historical_means + shrinkage_factor * historical_means.mean()
        expected_returns = shrunk_historical + factor_score * 0.4
        
    elif method == 'black_litterman':
        # Simplified Black-Litterman (equilibrium + views)
        cov_matrix = returns_matrix.cov() * 252
        
        # Market cap weights (equal for simplicity - replace with actual market caps)
        market_weights = np.ones(len(securities)) / len(securities)
        
        # Implied equilibrium returns
        equilibrium_returns = risk_aversion * cov_matrix.dot(market_weights)
        
        # Simple momentum view: recent outperformers will continue
        momentum_view = calculate_momentum_factors(returns_matrix)['momentum_63d']
        view_strength = 0.3
        
        expected_returns = equilibrium_returns + view_strength * momentum_view
        
    else:
        raise ValueError("Method must be one of: 'historical', 'shrinkage', 'momentum', 'mean_reversion', 'multi_factor', 'black_litterman'")
    
    # Ensure reasonable bounds (annual returns between -50% and +100%)
    expected_returns = expected_returns.clip(-0.5, 1.0)
    
    return expected_returns

def backtest_expected_returns(returns_matrix, forecast_horizon=21, test_periods=12):
    """
    Backtest different expected returns models
    
    Parameters:
    forecast_horizon: days ahead to forecast (21 = 1 month)
    test_periods: number of test periods
    """
    
    methods = ['historical', 'shrinkage', 'momentum', 'mean_reversion', 'multi_factor']
    results = {method: {'predictions': [], 'actuals': [], 'dates': []} for method in methods}
    
    print(f"Starting backtest with {len(returns_matrix.columns)} securities...")
    print(f"Data range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
    
    # Rolling backtest
    successful_tests = 0
    for i in range(test_periods):
        # Define training and test periods
        end_train = len(returns_matrix) - (test_periods - i) * forecast_horizon
        start_train = max(0, end_train - 252)  # 1 year of training data
        
        if start_train >= end_train or end_train + forecast_horizon > len(returns_matrix):
            print(f"Skipping test period {i+1}: insufficient data")
            continue
            
        # Training data
        train_data = returns_matrix.iloc[start_train:end_train]
        
        # Actual future returns
        future_data = returns_matrix.iloc[end_train:end_train + forecast_horizon]
        actual_returns = (1 + future_data).prod() - 1  # Cumulative returns over forecast horizon
        
        test_date = returns_matrix.index[end_train]
        print(f"Processing test period {i+1}: {test_date}")
        
        # Test each method
        for method in methods:
            try:
                expected_rets = build_factor_model_returns(train_data, method=method)
                # Scale to forecast horizon
                expected_rets_scaled = expected_rets * (forecast_horizon / 252)
                
                results[method]['predictions'].append(expected_rets_scaled)
                results[method]['actuals'].append(actual_returns)
                results[method]['dates'].append(test_date)
                
            except Exception as e:
                print(f"Error in method {method} for {test_date}: {e}")
                continue
        
        successful_tests += 1
    
    print(f"Completed {successful_tests} successful test periods")
    
    # Calculate performance metrics
    performance_metrics = {}
    
    for method in methods:
        if len(results[method]['predictions']) > 0:
            try:
                # Concatenate all predictions and actuals
                all_predictions = pd.concat(results[method]['predictions'], axis=1).mean(axis=1)
                all_actuals = pd.concat(results[method]['actuals'], axis=1).mean(axis=1)
                
                # Align securities
                common_securities = all_predictions.index.intersection(all_actuals.index)
                if len(common_securities) == 0:
                    print(f"No common securities for method {method}")
                    continue
                    
                pred_aligned = all_predictions[common_securities]
                actual_aligned = all_actuals[common_securities]
                
                # Remove any remaining NaN values
                valid_mask = pred_aligned.notna() & actual_aligned.notna()
                pred_clean = pred_aligned[valid_mask]
                actual_clean = actual_aligned[valid_mask]
                
                if len(pred_clean) < 5:  # Need at least 5 securities for meaningful metrics
                    print(f"Insufficient valid data for method {method}")
                    continue
                
                # Calculate metrics
                mse = mean_squared_error(actual_clean, pred_clean)
                correlation = pred_clean.corr(actual_clean)
                
                # Information Coefficient (rank correlation)
                ic = pred_clean.rank().corr(actual_clean.rank(), method='spearman')
                
                performance_metrics[method] = {
                    'MSE': mse,
                    'Correlation': correlation,
                    'Information_Coefficient': ic,
                    'RMSE': np.sqrt(mse),
                    'N_Securities': len(pred_clean)
                }
                
            except Exception as e:
                print(f"Error calculating metrics for {method}: {e}")
                continue
    
    return performance_metrics, results

def plot_expected_returns_comparison(returns_matrix, securities_subset=None):
    """
    Compare different expected returns models
    """
    
    methods = ['historical', 'shrinkage', 'momentum', 'mean_reversion', 'multi_factor']
    
    if securities_subset is None:
        securities_subset = returns_matrix.columns[:20]  # Top 20 for readability
    
    expected_returns_df = pd.DataFrame(index=securities_subset)
    
    for method in methods:
        expected_rets = build_factor_model_returns(returns_matrix, method=method)
        expected_returns_df[method] = expected_rets[securities_subset]
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Heatmap of expected returns
    sns.heatmap(expected_returns_df.T, annot=True, fmt='.2%', cmap='RdYlGn', 
                center=0, ax=axes[0, 0])
    axes[0, 0].set_title('Expected Returns by Method')
    axes[0, 0].set_xlabel('Securities')
    
    # Distribution of expected returns
    expected_returns_df.plot(kind='box', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Expected Returns')
    axes[0, 1].set_ylabel('Expected Return')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Correlation between methods
    corr_matrix = expected_returns_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Correlation Between Methods')
    
    # Scatter plot: Historical vs Multi-factor
    axes[1, 1].scatter(expected_returns_df['historical'], expected_returns_df['multi_factor'], 
                       alpha=0.7, s=50)
    axes[1, 1].plot([-0.3, 0.5], [-0.3, 0.5], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('Historical Mean Returns')
    axes[1, 1].set_ylabel('Multi-Factor Expected Returns')
    axes[1, 1].set_title('Historical vs Multi-Factor')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return expected_returns_df

def integrate_expected_returns_with_optimizer(returns_matrix, selected_securities, 
                                            expected_returns_method='multi_factor',
                                            **optimizer_kwargs):
    """
    Integrate expected returns models with the CVXPY optimizer
    
    Parameters:
    returns_matrix: DataFrame with historical returns
    selected_securities: list of securities for portfolio
    expected_returns_method: method for calculating expected returns
    **optimizer_kwargs: additional arguments for optimize_portfolio_cvxpy
    """
    
    # Filter returns matrix for selected securities
    portfolio_returns_matrix = returns_matrix[selected_securities]
    
    # Calculate ex-ante expected returns
    expected_returns = build_factor_model_returns(
        portfolio_returns_matrix, 
        method=expected_returns_method
    )
    
    # Calculate covariance matrix from historical data
    cov_matrix = portfolio_returns_matrix.cov().values * 252
    
    print(f"Expected Returns Model: {expected_returns_method}")
    print(f"Expected returns range: {expected_returns.min():.2%} to {expected_returns.max():.2%}")
    print(f"Expected returns mean: {expected_returns.mean():.2%}")
    
    return expected_returns.values, cov_matrix, selected_securities

# Assuming your dataframe is called 'final_portfolio_data'
def prepare_returns_matrix_safe(df, date_col='Date', ticker_col='Ticker', return_col='return'):
    """
    Convert long format dataframe to wide format returns matrix with error handling
    """
    
    # Check if columns exist
    required_cols = [date_col, ticker_col, return_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Check for missing return values
    print(f"Missing return values: {df[return_col].isna().sum()}")
    
    # Pivot to wide format
    try:
        returns_matrix = df.pivot_table(
            index=date_col,
            columns=ticker_col, 
            values=return_col,
            aggfunc='last'  # Take last value if multiple per day
        )
    except Exception as e:
        print(f"Pivot error: {e}")
        return None
    
    # Clean the data
    print(f"Matrix shape before cleaning: {returns_matrix.shape}")
    returns_matrix = returns_matrix.fillna(method='ffill')
    returns_matrix = returns_matrix.dropna(axis=1, thresh=len(returns_matrix)*0.5)  # Remove tickers with >50% missing
    returns_matrix = returns_matrix.dropna()
    
    print(f"Returns matrix shape: {returns_matrix.shape}")
    print(f"Date range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
    print(f"Number of tickers: {len(returns_matrix.columns)}")
    
    return returns_matrix


    

# Example usage:
"""
# Backtest different expected returns models
performance_metrics, backtest_results = backtest_expected_returns(
    returns_matrix=returns_matrix,
    forecast_horizon=21,  # 1 month ahead
    test_periods=12       # 12 test periods
)

print("Expected Returns Model Performance:")
for method, metrics in performance_metrics.items():
    print(f"{method:15} | IC: {metrics['Information_Coefficient']:.3f} | "
          f"Corr: {metrics['Correlation']:.3f} | RMSE: {metrics['RMSE']:.4f}")

# Compare different methods visually
expected_returns_comparison = plot_expected_returns_comparison(
    returns_matrix=returns_matrix,
    securities_subset=returns_matrix.columns[:15]  # Top 15 securities from returns_matrix
)

# Use the best performing method in optimization
best_method = max(performance_metrics.keys(), 
                 key=lambda x: performance_metrics[x]['Information_Coefficient'])
print(f"\\nBest performing method: {best_method}")

# Integrate with CVXPY optimizer
expected_returns, cov_matrix, securities = integrate_expected_returns_with_optimizer(
    returns_matrix=returns_matrix,
    selected_securities=selected_securities,
    expected_returns_method=best_method  # or 'multi_factor' for robust choice
)

# Now use in CVXPY optimization
optimal_weights, portfolio_return, portfolio_vol, sharpe_ratio, status = optimize_portfolio_cvxpy(
    mean_returns=expected_returns,  # Ex-ante expected returns!
    cov_matrix=cov_matrix,
    securities=securities,
    risk_aversion=1.0,
    l1_penalty=0.01,
    max_weight=0.30,
    min_weight=0.005
)
"""