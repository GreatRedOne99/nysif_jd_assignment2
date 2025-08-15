import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any

class BacktestResultsStorage:
    """
    Comprehensive storage system for monthly portfolio backtesting results
    """
    
    def __init__(self, start_date: str, end_date: str, benchmark_name: str = "SPY"):
        """
        Initialize the backtesting results storage
        
        Parameters:
        start_date: Start date of backtest ('YYYY-MM-DD')
        end_date: End date of backtest ('YYYY-MM-DD')  
        benchmark_name: Name of benchmark for comparison
        """
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_name = benchmark_name
        
        # Initialize storage containers
        self.monthly_metrics = []           # Monthly performance metrics
        self.portfolio_weights = []         # Monthly portfolio weights
        self.optimization_params = []       # Monthly optimization parameters
        self.prediction_accuracy = []       # XGBoost prediction accuracy
        self.risk_metrics = []              # Monthly risk metrics
        self.benchmark_comparison = []      # Monthly benchmark comparison
        self.transaction_costs = []         # Monthly transaction costs
        self.portfolio_characteristics = [] # Monthly portfolio characteristics
        
        # Cumulative tracking
        self.daily_returns = pd.DataFrame()     # Daily portfolio returns
        self.daily_positions = pd.DataFrame()   # Daily position values
        self.rebalance_dates = []              # Rebalancing dates
        
        print(f"Initialized backtest storage for {start_date} to {end_date}")
        print(f"Benchmark: {benchmark_name}")
    
    def store_monthly_results(self, 
                            rebalance_date: str,
                            portfolio_weights: pd.DataFrame,
                            optimization_metrics: Dict,
                            benchmark_metrics: Dict,
                            prediction_results: Dict = None,
                            transaction_costs: Dict = None,
                            additional_params: Dict = None):
        """
        Store all results from a single month's rebalancing
        
        Parameters:
        rebalance_date: Date of rebalancing ('YYYY-MM-DD')
        portfolio_weights: DataFrame with Security, Weight columns
        optimization_metrics: Dict with portfolio optimization results
        benchmark_metrics: Dict with benchmark comparison metrics
        prediction_results: Dict with XGBoost prediction accuracy (optional)
        transaction_costs: Dict with transaction cost analysis (optional)
        additional_params: Any additional parameters to store (optional)
        """
        
        rebalance_dt = pd.to_datetime(rebalance_date)
        self.rebalance_dates.append(rebalance_dt)
        
        # Store portfolio weights with metadata
        weights_record = {
            'date': rebalance_date,
            'weights': portfolio_weights.copy(),
            'num_positions': len(portfolio_weights),
            'max_weight': portfolio_weights['Weight'].max(),
            'top5_concentration': portfolio_weights.head(5)['Weight'].sum(),
            'diversification_ratio': 1 / np.sum(portfolio_weights['Weight']**2) if len(portfolio_weights) > 0 else 0,
            'effective_stocks': 1 / np.sum(portfolio_weights['Weight']**2) if len(portfolio_weights) > 0 else 0
        }
        self.portfolio_weights.append(weights_record)
        
        # Store optimization metrics
        opt_record = {
            'date': rebalance_date,
            'expected_return': optimization_metrics.get('expected_return', np.nan),
            'expected_volatility': optimization_metrics.get('expected_volatility', np.nan),
            'sharpe_ratio': optimization_metrics.get('sharpe_ratio', np.nan),
            'optimization_status': optimization_metrics.get('status', 'unknown'),
            'risk_aversion': optimization_metrics.get('risk_aversion', np.nan),
            'l1_penalty': optimization_metrics.get('l1_penalty', np.nan),
            'max_weight_constraint': optimization_metrics.get('max_weight', np.nan),
            'min_weight_threshold': optimization_metrics.get('min_weight', np.nan),
            'solver_time': optimization_metrics.get('solver_time', np.nan)
        }
        self.optimization_params.append(opt_record)
        
        # Store benchmark comparison metrics
        bench_record = {
            'date': rebalance_date,
            **benchmark_metrics  # Unpack all benchmark metrics
        }
        self.benchmark_comparison.append(bench_record)
        
        # Store prediction accuracy if provided
        if prediction_results:
            pred_record = {
                'date': rebalance_date,
                'total_predictions': prediction_results.get('total_predictions', 0),
                'accurate_predictions': prediction_results.get('accurate_predictions', 0),
                'accuracy_rate': prediction_results.get('accuracy_rate', np.nan),
                'precision_include': prediction_results.get('precision_include', np.nan),
                'recall_include': prediction_results.get('recall_include', np.nan),
                'f1_score': prediction_results.get('f1_score', np.nan),
                'feature_importance': prediction_results.get('feature_importance', {})
            }
            self.prediction_accuracy.append(pred_record)
        
        # Store transaction costs if provided
        if transaction_costs:
            cost_record = {
                'date': rebalance_date,
                'total_turnover': transaction_costs.get('total_turnover', 0),
                'transaction_costs': transaction_costs.get('transaction_costs', 0),
                'cost_bps': transaction_costs.get('cost_bps', 0),
                'num_trades': transaction_costs.get('num_trades', 0),
                'new_positions': transaction_costs.get('new_positions', 0),
                'closed_positions': transaction_costs.get('closed_positions', 0)
            }
            self.transaction_costs.append(cost_record)
        
        # Store portfolio characteristics
        char_record = {
            'date': rebalance_date,
            'portfolio_beta': benchmark_metrics.get('Beta', np.nan),
            'portfolio_alpha': benchmark_metrics.get('Alpha_Annual', np.nan),
            'tracking_error': benchmark_metrics.get('Tracking_Error', np.nan),
            'information_ratio': benchmark_metrics.get('Information_Ratio', np.nan),
            'up_capture': benchmark_metrics.get('Up_Capture_Ratio', np.nan),
            'down_capture': benchmark_metrics.get('Down_Capture_Ratio', np.nan),
            **(additional_params or {})
        }
        self.portfolio_characteristics.append(char_record)
        
        print(f"Stored results for {rebalance_date}: {len(portfolio_weights)} positions, "
              f"Expected Return: {optimization_metrics.get('expected_return', np.nan):.2%}")
    
    def store_daily_performance(self, daily_returns_series: pd.Series, 
                                      daily_benchmark_returns: pd.Series,
                                      daily_positions: pd.DataFrame = None):
        """
        Store daily performance data
        
        Parameters:
        daily_returns_series: Series with date index and portfolio returns
        daily_benchmark_returns: Series with date index and benchmark returns  
        daily_positions: DataFrame with daily position values (optional)
        """
        
        # Store daily returns
        if self.daily_returns.empty:
            self.daily_returns = pd.DataFrame({
                'portfolio_return': daily_returns_series,
                'benchmark_return': daily_benchmark_returns
            })
        else:
            new_returns = pd.DataFrame({
                'portfolio_return': daily_returns_series,
                'benchmark_return': daily_benchmark_returns
            })
            self.daily_returns = pd.concat([self.daily_returns, new_returns]).drop_duplicates()
        
        # Store daily positions if provided
        if daily_positions is not None:
            if self.daily_positions.empty:
                self.daily_positions = daily_positions.copy()
            else:
                self.daily_positions = pd.concat([self.daily_positions, daily_positions]).drop_duplicates()
    
    def get_summary_dataframes(self):
        """
        Convert stored results to pandas DataFrames for analysis
        
        Returns:
        Dictionary of DataFrames with all stored results
        """
        
        results = {}
        
        # Monthly metrics summary
        if self.optimization_params:
            results['monthly_optimization'] = pd.DataFrame(self.optimization_params)
            results['monthly_optimization']['date'] = pd.to_datetime(results['monthly_optimization']['date'])
            results['monthly_optimization'].set_index('date', inplace=True)
        
        # Benchmark comparison summary  
        if self.benchmark_comparison:
            results['benchmark_comparison'] = pd.DataFrame(self.benchmark_comparison)
            results['benchmark_comparison']['date'] = pd.to_datetime(results['benchmark_comparison']['date'])
            results['benchmark_comparison'].set_index('date', inplace=True)
        
        # Portfolio characteristics summary
        if self.portfolio_characteristics:
            results['portfolio_characteristics'] = pd.DataFrame(self.portfolio_characteristics)
            results['portfolio_characteristics']['date'] = pd.to_datetime(results['portfolio_characteristics']['date'])
            results['portfolio_characteristics'].set_index('date', inplace=True)
        
        # Portfolio weights summary (aggregated metrics)
        if self.portfolio_weights:
            weights_summary = []
            for record in self.portfolio_weights:
                weights_summary.append({
                    'date': record['date'],
                    'num_positions': record['num_positions'],
                    'max_weight': record['max_weight'],
                    'top5_concentration': record['top5_concentration'],
                    'diversification_ratio': record['diversification_ratio'],
                    'effective_stocks': record['effective_stocks']
                })
            results['portfolio_composition'] = pd.DataFrame(weights_summary)
            results['portfolio_composition']['date'] = pd.to_datetime(results['portfolio_composition']['date'])
            results['portfolio_composition'].set_index('date', inplace=True)
        
        # Prediction accuracy summary
        if self.prediction_accuracy:
            results['prediction_accuracy'] = pd.DataFrame(self.prediction_accuracy)
            results['prediction_accuracy']['date'] = pd.to_datetime(results['prediction_accuracy']['date'])
            results['prediction_accuracy'].set_index('date', inplace=True)
        
        # Transaction costs summary
        if self.transaction_costs:
            results['transaction_costs'] = pd.DataFrame(self.transaction_costs)
            results['transaction_costs']['date'] = pd.to_datetime(results['transaction_costs']['date'])
            results['transaction_costs'].set_index('date', inplace=True)
        
        # Daily returns
        # if not self.daily_returns.empty:
        #     results['daily_returns'] = self.daily_returns.copy()
        if not self.daily_returns.empty:
            results['daily_returns'] = self.daily_returns['portfolio_return'].copy()  # Portfolio returns only
            results['daily_benchmark_returns'] = self.daily_returns['benchmark_return'].copy()  # Benchmark returns only
            results['daily_returns_combined'] = self.daily_returns.copy()  # Both together
            
        # Daily positions
        if not self.daily_positions.empty:
            results['daily_positions'] = self.daily_positions.copy()
        
        return results
    
    def calculate_backtest_summary_stats(self):
        """
        Calculate overall backtest summary statistics
        """
        
        if self.daily_returns.empty:
            print("No daily returns data available for summary statistics")
            return None
        
        portfolio_returns = self.daily_returns['portfolio_return'].dropna()
        benchmark_returns = self.daily_returns['benchmark_return'].dropna()
        
        # Align returns
        aligned_returns = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if aligned_returns.empty:
            print("No aligned return data available")
            return None
        
        portfolio_ret = aligned_returns['portfolio']
        benchmark_ret = aligned_returns['benchmark']
        
        # Calculate summary metrics
        periods_per_year = 252
        
        summary_stats = {
            'backtest_period': f"{self.start_date} to {self.end_date}",
            'total_trading_days': len(portfolio_ret),
            'num_rebalances': len(self.rebalance_dates),
            
            # Returns
            'total_portfolio_return': (1 + portfolio_ret).prod() - 1,
            'total_benchmark_return': (1 + benchmark_ret).prod() - 1,
            'annualized_portfolio_return': (1 + portfolio_ret).prod() ** (periods_per_year / len(portfolio_ret)) - 1,
            'annualized_benchmark_return': (1 + benchmark_ret).prod() ** (periods_per_year / len(benchmark_ret)) - 1,
            
            # Risk metrics
            'portfolio_volatility': portfolio_ret.std() * np.sqrt(periods_per_year),
            'benchmark_volatility': benchmark_ret.std() * np.sqrt(periods_per_year),
            'portfolio_sharpe': (portfolio_ret.mean() * periods_per_year - 0.02) / (portfolio_ret.std() * np.sqrt(periods_per_year)),
            'benchmark_sharpe': (benchmark_ret.mean() * periods_per_year - 0.02) / (benchmark_ret.std() * np.sqrt(periods_per_year)),
            
            # Relative metrics
            'excess_return': portfolio_ret.mean() - benchmark_ret.mean(),
            'tracking_error': (portfolio_ret - benchmark_ret).std() * np.sqrt(periods_per_year),
            'information_ratio': (portfolio_ret.mean() - benchmark_ret.mean()) / (portfolio_ret - benchmark_ret).std(),
            
            # Drawdown
            'portfolio_max_drawdown': self._calculate_max_drawdown(portfolio_ret),
            'benchmark_max_drawdown': self._calculate_max_drawdown(benchmark_ret),
            
            # Win rates
            'monthly_win_rate': (portfolio_ret > benchmark_ret).mean(),
            'positive_months': (portfolio_ret > 0).mean()
        }
        
        return summary_stats
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def save_results(self, filepath_prefix: str):
        """
        Save all results to files
        
        Parameters:
        filepath_prefix: Prefix for saved files (e.g., 'backtest_results_2023')
        """
        
        # Save summary DataFrames as CSV
        summary_dfs = self.get_summary_dataframes()
        for name, df in summary_dfs.items():
            if not df.empty:
                df.to_csv(f"{filepath_prefix}_{name}.csv")
        
        # Save individual portfolio weights as pickle for detailed analysis
        with open(f"{filepath_prefix}_portfolio_weights.pkl", 'wb') as f:
            pickle.dump(self.portfolio_weights, f)
        
        # Save backtest configuration and summary stats
        config = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'benchmark_name': self.benchmark_name,
            'rebalance_dates': [d.strftime('%Y-%m-%d') for d in self.rebalance_dates],
            'summary_stats': self.calculate_backtest_summary_stats()
        }
        
        with open(f"{filepath_prefix}_config_and_summary.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"Saved backtest results with prefix: {filepath_prefix}")
        print(f"Files created: {len(summary_dfs)} CSV files + 2 additional files")
    
    def load_results(self, filepath_prefix: str):
        """
        Load previously saved results
        
        Parameters:
        filepath_prefix: Prefix of saved files to load
        """
        
        try:
            # Load configuration
            with open(f"{filepath_prefix}_config_and_summary.json", 'r') as f:
                config = json.load(f)
                self.start_date = config['start_date']
                self.end_date = config['end_date'] 
                self.benchmark_name = config['benchmark_name']
                self.rebalance_dates = [pd.to_datetime(d) for d in config['rebalance_dates']]
            
            # Load portfolio weights
            with open(f"{filepath_prefix}_portfolio_weights.pkl", 'rb') as f:
                self.portfolio_weights = pickle.load(f)
            
            # Load other data from CSVs
            try:
                self.daily_returns = pd.read_csv(f"{filepath_prefix}_daily_returns.csv", index_col=0, parse_dates=True)
            except FileNotFoundError:
                pass
            
            print(f"Loaded backtest results from: {filepath_prefix}")
            
        except FileNotFoundError as e:
            print(f"Could not load results: {e}")

# Helper functions for integration with your existing code

def calculate_transaction_costs(old_weights: pd.DataFrame, new_weights: pd.DataFrame, 
                               cost_per_trade_bps: float = 5.0):
    """
    Calculate transaction costs between two portfolio weight allocations
    
    Parameters:
    old_weights: Previous period portfolio weights
    new_weights: New period portfolio weights  
    cost_per_trade_bps: Transaction cost in basis points per trade
    """
    
    if old_weights is None or old_weights.empty:
        # First period - assume we're buying everything
        total_turnover = new_weights['Weight'].sum()
        num_trades = len(new_weights)
        transaction_costs = total_turnover * (cost_per_trade_bps / 10000)
        
        return {
            'total_turnover': total_turnover,
            'transaction_costs': transaction_costs,
            'cost_bps': cost_per_trade_bps,
            'num_trades': num_trades,
            'new_positions': len(new_weights),
            'closed_positions': 0
        }
    
    # Merge old and new weights
    old_weights_dict = dict(zip(old_weights['Security'], old_weights['Weight']))
    new_weights_dict = dict(zip(new_weights['Security'], new_weights['Weight']))
    
    all_securities = set(old_weights_dict.keys()) | set(new_weights_dict.keys())
    
    turnover = 0
    trades = 0
    new_positions = 0
    closed_positions = 0
    
    for security in all_securities:
        old_weight = old_weights_dict.get(security, 0)
        new_weight = new_weights_dict.get(security, 0)
        
        weight_change = abs(new_weight - old_weight)
        turnover += weight_change
        
        if weight_change > 0.001:  # Only count significant changes as trades
            trades += 1
        
        if old_weight == 0 and new_weight > 0:
            new_positions += 1
        elif old_weight > 0 and new_weight == 0:
            closed_positions += 1
    
    transaction_costs = turnover * (cost_per_trade_bps / 10000)
    
    return {
        'total_turnover': turnover,
        'transaction_costs': transaction_costs,
        'cost_bps': cost_per_trade_bps,
        'num_trades': trades,
        'new_positions': new_positions,
        'closed_positions': closed_positions
    }

# Example integration with your existing backtest loop:
"""
# Initialize storage
backtest_storage = BacktestResultsStorage(
    start_date='2022-01-01',
    end_date='2024-01-01', 
    benchmark_name='SPY'
)

# Your monthly backtest loop
for month_date in monthly_rebalance_dates:
    
    # Your existing code to get predictions and optimize portfolio
    # ... XGBoost predictions ...
    # ... CVXPY optimization ...
    
    # Store optimization metrics
    optimization_metrics = {
        'expected_return': portfolio_return,
        'expected_volatility': portfolio_vol,
        'sharpe_ratio': sharpe_ratio,
        'status': status,
        'risk_aversion': risk_aversion,
        'l1_penalty': l1_penalty,
        'max_weight': max_weight,
        'min_weight': min_weight
    }
    
    # Calculate transaction costs vs previous month
    transaction_costs = calculate_transaction_costs(previous_weights, optimal_weights)
    
    # Store all monthly results
    backtest_storage.store_monthly_results(
        rebalance_date=month_date,
        portfolio_weights=optimal_weights,
        optimization_metrics=optimization_metrics,
        benchmark_metrics=comparison_metrics,
        prediction_results=prediction_accuracy_dict,  # Your XGBoost accuracy results
        transaction_costs=transaction_costs
    )
    
    # Store daily performance for this month
    backtest_storage.store_daily_performance(
        daily_returns_series=monthly_portfolio_returns,
        daily_benchmark_returns=monthly_benchmark_returns
    )
    
    previous_weights = optimal_weights.copy()

# After backtest loop completes
summary_stats = backtest_storage.calculate_backtest_summary_stats()
print("Backtest Summary:")
for key, value in summary_stats.items():
    print(f"{key}: {value}")

# Save all results
backtest_storage.save_results('portfolio_backtest_2022_2024')

# Get DataFrames for further analysis
results_dfs = backtest_storage.get_summary_dataframes()
monthly_performance = results_dfs['benchmark_comparison']
portfolio_composition = results_dfs['portfolio_composition']
"""