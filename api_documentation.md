# API Documentation - MTUM Replica Portfolio System

## Table of Contents
1. [Module Overview](#module-overview)
2. [Stage 1: Portfolio Construction Modules](#stage-1-portfolio-construction-modules)
3. [Stage 2: Analysis & Reporting Module](#stage-2-analysis--reporting-module)
4. [XGBoost Portfolio Model API](#xgboost-portfolio-model-api)
5. [Expected Returns Models API](#expected-returns-models-api)
6. [Portfolio Optimization API](#portfolio-optimization-api)
7. [Backtest Storage API](#backtest-storage-api)
8. [Performance Analysis API](#performance-analysis-api)
9. [MTUM Analysis Toolkit API](#mtum-analysis-toolkit-api)
10. [Utility Functions](#utility-functions)
11. [Error Handling](#error-handling)
12. [Configuration Reference](#configuration-reference)

## Module Overview

The MTUM Replica Portfolio System consists of **seven main modules** organized into two stages:

### Stage 1: Portfolio Construction Pipeline
```python
# Core portfolio construction modules
from xgb_portfolio_model import (
    train_xgb_portfolio_model,
    predict_next_month_portfolio,
    plot_model_performance
)

from expected_returns_models import (
    build_factor_model_returns,
    backtest_expected_returns,
    integrate_expected_returns_with_optimizer
)

from portfolio_optimization import (
    optimize_portfolio_cvxpy,
    prepare_cvxpy_data,
    calculate_benchmark_comparison_metrics
)

from backtest_storage import (
    BacktestResultsStorage,
    calculate_transaction_costs
)

from portfolio_performance_summary import (
    generate_portfolio_performance_report,
    calculate_key_statistics
)

from portfolio_precleaning import (
    clean_data_pipeline_step,
    prepare_returns_matrix_safe
)
```

### Stage 2: Professional Analysis & Reporting
```python
# Analysis and reporting toolkit
import mtum_analysis_toolkit as mtum

# Core functions
mtum.load_all_data()
mtum.generate_executive_summary(data)
mtum.plot_cumulative_performance(data)
mtum.generate_markdown_executive_report(data, summary_metrics)
```

## Stage 1: Portfolio Construction Modules

### Core Architecture
The Stage 1 modules work together to create and backtest portfolios:

```
Raw Data → Feature Engineering → ML Prediction → Portfolio Optimization → Performance Tracking
```

---

## Stage 2: Analysis & Reporting Module

### MTUM Analysis Toolkit Overview
**Module**: `mtum_analysis_toolkit.py`  
**Purpose**: Complete post-analysis toolkit for professional reporting and business intelligence

**Key Features**:
- 🔍 **Automatic Data Loading**: Finds and loads latest backtest results
- 📊 **Professional Visualizations**: Institutional-quality charts and analysis
- 📋 **Executive Reporting**: Automated markdown and Excel report generation
- 💼 **Business Intelligence**: Interview-ready talking points and conclusions

---

## XGBoost Portfolio Model API

### Core Functions

#### `train_xgb_portfolio_model()`

```python
def train_xgb_portfolio_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    date_col: str = 'Date',
    security_col: str = 'Ticker',
    target_col: str = 'target',
    test_size: float = 0.2
) -> Tuple[XGBClassifier, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    """
    Train XGBoost model to predict next month's portfolio inclusion.
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical data with features and targets
    feature_columns : List[str]
        List of feature column names for model training
    date_col : str, default 'Date'
        Date column name for time-series handling
    security_col : str, default 'Ticker'
        Security identifier column name
    target_col : str, default 'target'
        Binary target variable (1=include, 0=exclude)
    test_size : float, default 0.2
        Fraction of data for test set validation
        
    Returns
    -------
    model : XGBClassifier
        Trained XGBoost model with optimized hyperparameters
    feature_importance : pd.DataFrame
        Feature importance scores from the trained model
    processed_data : pd.DataFrame
        Processed training data with all features
    y_test : np.ndarray
        Test set true labels for validation
    y_test_prob : np.ndarray
        Test set predicted probabilities
    prediction_results : Dict
        Performance metrics including AUC, precision, recall, F1-score
        
    Examples
    --------
    >>> model, importance, data, y_test, y_prob, results = train_xgb_portfolio_model(
    ...     df=historical_data,
    ...     feature_columns=['mom_1m', 'mom_3m', 'volatility_20d'],
    ...     target_col='include_next_month'
    ... )
    >>> print(f"Model AUC: {results['auc']:.3f}")
    >>> print(f"Top feature: {importance.iloc[0]['Feature']}")
    """
```

#### `predict_next_month_portfolio()`

```python
def predict_next_month_portfolio(
    model: XGBClassifier,
    current_data: pd.DataFrame,
    feature_columns: List[str],
    probability_threshold: float = 0.5
) -> Tuple[List[str], pd.DataFrame]:
    """
    Predict securities for next month's portfolio using trained model.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model
    current_data : pd.DataFrame
        Current period data with all features
    feature_columns : List[str]
        Feature columns used in model training
    probability_threshold : float, default 0.5
        Minimum probability for portfolio inclusion
        
    Returns
    -------
    selected_securities : List[str]
        List of securities predicted for inclusion
    prediction_details : pd.DataFrame
        Detailed predictions with probabilities for all securities
    """
```

#### `plot_model_performance()`

```python
def plot_model_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    feature_importance: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive model performance visualization.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    feature_importance : pd.DataFrame
        Feature importance DataFrame
    save_path : Optional[str], default None
        Path to save the plot
        
    Side Effects
    ------------
    Creates multi-panel visualization showing:
    1. ROC curve with AUC score
    2. Precision-Recall curve
    3. Feature importance ranking
    4. Prediction probability distribution
    """
```

---

## Expected Returns Models API

### Core Functions

#### `build_factor_model_returns()`

```python
def build_factor_model_returns(
    returns_matrix: pd.DataFrame,
    method: str = 'multi_factor',
    shrinkage_factor: float = 0.3,
    risk_aversion: float = 3.0,
    lookback_periods: int = 252
) -> Tuple[pd.Series, Dict]:
    """
    Build ex-ante expected returns using factor models.
    
    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Wide format returns matrix (dates x securities)
    method : str, default 'multi_factor'
        Expected returns method: {'historical', 'momentum', 'shrinkage', 'multi_factor'}
    shrinkage_factor : float, default 0.3
        Shrinkage intensity for James-Stein estimator (0=sample mean, 1=grand mean)
    risk_aversion : float, default 3.0
        Risk aversion parameter for mean reversion component
    lookback_periods : int, default 252
        Number of historical periods for calculation
        
    Returns
    -------
    expected_returns : pd.Series
        Expected returns for each security (annualized)
    model_details : Dict
        Model components and diagnostics:
        - 'momentum_component': Momentum-based expected returns
        - 'mean_reversion_component': Mean reversion expected returns
        - 'historical_mean': Historical average returns
        - 'shrinkage_intensity': Applied shrinkage factor
        - 'method_used': Actual method applied
        
    Examples
    --------
    >>> returns_matrix = prepare_returns_matrix(daily_data)
    >>> expected_rets, details = build_factor_model_returns(
    ...     returns_matrix=returns_matrix,
    ...     method='multi_factor',
    ...     shrinkage_factor=0.3
    ... )
    >>> print(f"Expected return range: {expected_rets.min():.1%} to {expected_rets.max():.1%}")
    """
```

#### `backtest_expected_returns()`

```python
def backtest_expected_returns(
    returns_matrix: pd.DataFrame,
    method: str = 'multi_factor',
    lookback_window: int = 252,
    forecast_horizon: int = 21
) -> Tuple[pd.DataFrame, Dict]:
    """
    Backtest expected returns model performance over time.
    
    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Historical returns matrix for backtesting
    method : str, default 'multi_factor'
        Expected returns method to test
    lookback_window : int, default 252
        Rolling window size for model estimation
    forecast_horizon : int, default 21
        Forward-looking period for validation (trading days)
        
    Returns
    -------
    backtest_results : pd.DataFrame
        Time series of forecast accuracy metrics
    summary_stats : Dict
        Overall performance statistics:
        - 'hit_rate': Directional accuracy
        - 'rank_correlation': Spearman correlation with realized returns
        - 'mse': Mean squared error
        - 'mae': Mean absolute error
    """
```

---

## Portfolio Optimization API

### Core Functions

#### `optimize_portfolio_cvxpy()`

```python
def optimize_portfolio_cvxpy(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
    l1_penalty: float = 0.01,
    max_weight: float = 0.30,
    min_weight: float = 0.005,
    max_positions: int = 20,
    benchmark_weights: Optional[pd.Series] = None
) -> Tuple[pd.Series, Dict]:
    """
    Optimize portfolio weights using CVXPY with L1 regularization.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns for each security (annualized)
    covariance_matrix : pd.DataFrame
        Covariance matrix of returns (annualized)
    risk_aversion : float, default 1.0
        Risk aversion parameter (higher = more conservative)
    l1_penalty : float, default 0.01
        L1 regularization strength (higher = more sparse)
    max_weight : float, default 0.30
        Maximum weight constraint per security
    min_weight : float, default 0.005
        Minimum meaningful weight threshold
    max_positions : int, default 20
        Maximum number of positions in portfolio
    benchmark_weights : Optional[pd.Series], default None
        Benchmark weights for tracking error minimization
        
    Returns
    -------
    optimal_weights : pd.Series
        Optimal portfolio weights (sum to 1.0)
    optimization_results : Dict
        Optimization diagnostics:
        - 'expected_return': Portfolio expected return
        - 'expected_volatility': Portfolio expected volatility
        - 'sharpe_ratio': Expected Sharpe ratio
        - 'num_positions': Number of non-zero positions
        - 'status': Solver status
        - 'solve_time': Optimization time in seconds
        - 'objective_value': Final objective function value
        
    Examples
    --------
    >>> weights, results = optimize_portfolio_cvxpy(
    ...     expected_returns=monthly_expected_returns,
    ...     covariance_matrix=rolling_cov_matrix,
    ...     risk_aversion=1.0,
    ...     l1_penalty=0.02
    ... )
    >>> print(f"Portfolio expected return: {results['expected_return']:.1%}")
    >>> print(f"Number of positions: {results['num_positions']}")
    """
```

#### `calculate_benchmark_comparison_metrics()`

```python
def calculate_benchmark_comparison_metrics(
    portfolio_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    Calculate comprehensive portfolio vs benchmark comparison metrics.
    
    Parameters
    ----------
    portfolio_weights : pd.Series
        Current portfolio weights
    portfolio_returns : pd.Series
        Historical portfolio returns (daily)
    benchmark_returns : pd.Series
        Historical benchmark returns (daily)
    risk_free_rate : float, default 0.02
        Annual risk-free rate for Sharpe ratio calculation
        
    Returns
    -------
    comparison_metrics : Dict
        Comprehensive performance metrics:
        - 'portfolio_return': Annualized portfolio return
        - 'benchmark_return': Annualized benchmark return
        - 'excess_return': Portfolio excess return vs benchmark
        - 'tracking_error': Annualized tracking error
        - 'information_ratio': Information ratio (excess return / tracking error)
        - 'portfolio_sharpe': Portfolio Sharpe ratio
        - 'benchmark_sharpe': Benchmark Sharpe ratio
        - 'beta': Portfolio beta relative to benchmark
        - 'alpha': Portfolio alpha (annualized)
        - 'correlation': Correlation with benchmark
        - 'portfolio_volatility': Portfolio volatility (annualized)
        - 'benchmark_volatility': Benchmark volatility (annualized)
        - 'max_drawdown': Maximum drawdown period
        - 'calmar_ratio': Calmar ratio (return/max_drawdown)
    """
```

---

## Backtest Storage API

### Core Classes

#### `BacktestResultsStorage`

```python
class BacktestResultsStorage:
    """
    Comprehensive storage and management system for backtest results.
    
    Attributes
    ----------
    start_date : str
        Backtest start date ('YYYY-MM-DD')
    end_date : str
        Backtest end date ('YYYY-MM-DD')
    benchmark_name : str
        Benchmark name for comparison
    results_prefix : str
        Unique prefix for all result files
    monthly_results : List[Dict]
        Storage for monthly rebalancing results
    daily_performance : pd.DataFrame
        Daily portfolio and benchmark performance
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        benchmark_name: str = "MTUM"
    ):
        """
        Initialize backtest results storage.
        
        Parameters
        ----------
        start_date : str
            Backtest start date ('YYYY-MM-DD')
        end_date : str
            Backtest end date ('YYYY-MM-DD')
        benchmark_name : str, default "MTUM"
            Benchmark name for comparison
        """
```

#### Key Methods

```python
def store_monthly_results(
    self,
    rebalance_date: str,
    portfolio_weights: pd.DataFrame,
    optimization_metrics: Dict,
    benchmark_metrics: Dict,
    prediction_results: Optional[Dict] = None,
    transaction_costs: Optional[Dict] = None,
    additional_params: Optional[Dict] = None
) -> None:
    """
    Store comprehensive results from a single month's rebalancing.
    
    Parameters
    ----------
    rebalance_date : str
        Rebalancing date ('YYYY-MM-DD')
    portfolio_weights : pd.DataFrame
        Portfolio weights with columns ['Security', 'Weight']
    optimization_metrics : Dict
        Optimization results from optimize_portfolio_cvxpy()
    benchmark_metrics : Dict
        Benchmark comparison metrics from calculate_benchmark_comparison_metrics()
    prediction_results : Optional[Dict], default None
        XGBoost prediction accuracy metrics
    transaction_costs : Optional[Dict], default None
        Transaction cost analysis from calculate_transaction_costs()
    additional_params : Optional[Dict], default None
        Any additional parameters to store
    """

def get_summary_dataframes(self) -> Dict[str, pd.DataFrame]:
    """
    Convert stored results to analysis-ready DataFrames.
    
    Returns
    -------
    summary_dataframes : Dict[str, pd.DataFrame]
        Analysis-ready DataFrames:
        - 'benchmark_comparison': Monthly performance comparison metrics
        - 'portfolio_composition': Monthly portfolio characteristics
        - 'portfolio_weights': Historical portfolio weights
        - 'transaction_costs': Monthly transaction cost analysis
        - 'prediction_accuracy': ML model performance over time
        - 'daily_returns': Daily portfolio and benchmark returns
    """

def save_all_results(
    self,
    bulk_mode: bool = False,
    save_charts: bool = True,
    save_excel: bool = True,
    save_pickles: bool = True
) -> str:
    """
    Save all results to organized directory structure.
    
    Parameters
    ----------
    bulk_mode : bool, default False
        Use optimized bulk saving for faster processing
    save_charts : bool, default True
        Generate and save performance charts
    save_excel : bool, default True
        Save Excel summary files
    save_pickles : bool, default True
        Save pickle files for Python objects
        
    Returns
    -------
    results_directory : str
        Path to the organized results directory
    """
```

### Utility Functions

#### `calculate_transaction_costs()`

```python
def calculate_transaction_costs(
    old_weights: Optional[pd.DataFrame],
    new_weights: pd.DataFrame,
    cost_per_trade_bps: float = 5.0
) -> Dict:
    """
    Calculate transaction costs from portfolio rebalancing.
    
    Parameters
    ----------
    old_weights : Optional[pd.DataFrame]
        Previous period weights (None for first period)
    new_weights : pd.DataFrame
        New period weights with columns ['Security', 'Weight']
    cost_per_trade_bps : float, default 5.0
        Transaction cost in basis points per trade
        
    Returns
    -------
    cost_metrics : Dict
        Transaction cost breakdown:
        - 'total_turnover': Total portfolio turnover (sum of absolute weight changes)
        - 'transaction_costs': Total transaction costs in basis points
        - 'cost_bps': Cost per basis point of turnover
        - 'num_trades': Number of securities traded
        - 'new_positions': Number of newly opened positions
        - 'closed_positions': Number of closed positions
        - 'weight_changes': DataFrame of individual security weight changes
    """
```

---

## Performance Analysis API

### Core Functions

#### `generate_portfolio_performance_report()`

```python
def generate_portfolio_performance_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    feature_importance_df: Optional[pd.DataFrame] = None,
    portfolio_weights_history: Optional[pd.DataFrame] = None,
    portfolio_name: str = "Replica",
    benchmark_name: str = "MTUM"
) -> Tuple[Dict, pd.DataFrame]:
    """
    Generate complete portfolio performance report with statistics and visuals.
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio daily returns with datetime index
    benchmark_returns : pd.Series
        Benchmark daily returns with datetime index
    feature_importance_df : Optional[pd.DataFrame], default None
        XGBoost feature importance for visualization
    portfolio_weights_history : Optional[pd.DataFrame], default None
        Historical weights for turnover calculation
    portfolio_name : str, default "Replica"
        Name of the portfolio for labeling
    benchmark_name : str, default "MTUM"
        Name of the benchmark for labeling
        
    Returns
    -------
    key_statistics : Dict
        Comprehensive performance statistics
    aligned_data : pd.DataFrame
        Aligned portfolio and benchmark returns for further analysis
        
    Side Effects
    ------------
    - Prints formatted performance statistics table
    - Displays comprehensive performance visualization charts
    - Saves performance chart as PNG file if specified
    """
```

#### `calculate_key_statistics()`

```python
def calculate_key_statistics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_weights_history: Optional[pd.DataFrame] = None
) -> Tuple[Dict, pd.DataFrame]:
    """
    Calculate key portfolio statistics including tracking error and turnover.
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio daily returns
    benchmark_returns : pd.Series
        Benchmark daily returns
    portfolio_weights_history : Optional[pd.DataFrame], default None
        Historical portfolio weights for turnover calculation
        
    Returns
    -------
    statistics : Dict
        Key performance statistics:
        - 'annual_return': Annualized return
        - 'annual_volatility': Annualized volatility
        - 'sharpe_ratio': Sharpe ratio
        - 'max_drawdown': Maximum drawdown
        - 'tracking_error': Tracking error vs benchmark
        - 'information_ratio': Information ratio
        - 'beta': Beta relative to benchmark
        - 'alpha': Alpha (annualized excess return)
        - 'correlation': Correlation with benchmark
        - 'hit_rate': Percentage of days outperforming benchmark
        - 'average_turnover': Average monthly turnover (if weights provided)
    aligned_data : pd.DataFrame
        Aligned returns data for additional analysis
    """
```

---

## MTUM Analysis Toolkit API

### Module Overview

**Module**: `mtum_analysis_toolkit.py`  
**Version**: 1.0.0  
**Purpose**: Professional post-analysis toolkit for comprehensive portfolio analysis and reporting

### Core Workflow Functions

#### `load_all_data()`

```python
def load_all_data() -> Dict[str, pd.DataFrame]:
    """
    Automatically find and load all backtest results from latest directory.
    
    Returns
    -------
    data : Dict[str, pd.DataFrame]
        Dictionary containing all loaded datasets:
        - 'benchmark_comparison': Monthly performance comparison metrics
        - 'daily_returns': Daily portfolio and benchmark returns
        - 'portfolio_composition': Monthly portfolio characteristics
        - 'portfolio_weights': Historical portfolio weights
        - 'transaction_costs': Monthly transaction cost analysis
        - 'prediction_accuracy': ML model performance over time
        
    Examples
    --------
    >>> data = load_all_data()
    >>> print(f"Loaded datasets: {list(data.keys())}")
    >>> print(f"Analysis covers {len(data['benchmark_comparison'])} months")
    """
```

#### `generate_executive_summary()`

```python
def generate_executive_summary(data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Generate comprehensive executive summary with key metrics and analysis period.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data from load_all_data()
        
    Returns
    -------
    summary_metrics : Dict
        Executive summary containing:
        - 'start_date': Analysis start date
        - 'end_date': Analysis end date
        - 'period_years': Analysis period in years
        - 'period_days': Analysis period in days
        - 'portfolio_return': Annualized portfolio return
        - 'benchmark_return': Annualized benchmark return
        - 'excess_return': Excess return vs benchmark
        - 'tracking_error': Tracking error
        - 'info_ratio': Information ratio
        - 'sharpe_ratio': Portfolio Sharpe ratio
        - 'beta': Portfolio beta
        - 'alpha': Portfolio alpha
        - 'avg_holdings': Average number of holdings
        - 'concentration': Average top 5 concentration
        - 'avg_turnover': Average monthly turnover
        - 'avg_cost_bps': Average cost in basis points
        - 'model_accuracy': ML model accuracy
        - 'model_f1': ML model F1 score
        
    Side Effects
    ------------
    Prints comprehensive executive summary to console with:
    - Analysis period details
    - Key performance metrics
    - Portfolio characteristics
    - ML model performance
    """
```

### Visualization Functions

#### `plot_cumulative_performance()`

```python
def plot_cumulative_performance(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create cumulative performance comparison chart.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data containing 'daily_returns'
        
    Side Effects
    ------------
    Creates and displays professional performance chart showing:
    - Cumulative returns for portfolio vs benchmark
    - Performance metrics annotations
    - Period highlighting for key events
    - Professional styling for presentations
    """
```

#### `plot_portfolio_characteristics()`

```python
def plot_portfolio_characteristics(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create portfolio characteristics evolution chart.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data containing 'portfolio_composition' and 'benchmark_comparison'
        
    Side Effects
    ------------
    Creates multi-panel chart showing:
    - Number of holdings over time
    - Portfolio concentration metrics
    - Tracking error evolution
    - Information ratio development
    """
```

#### `plot_portfolio_weights()`

```python
def plot_portfolio_weights(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create portfolio weights allocation visualization.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data containing 'portfolio_weights'
        
    Side Effects
    ------------
    Creates weight allocation charts showing:
    - Current portfolio weights (pie chart)
    - Historical weight evolution (stacked area chart)
    - Individual security weight trends
    - Concentration analysis
    """
```

#### `plot_model_performance()`

```python
def plot_model_performance(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create ML model performance visualization.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data containing 'prediction_accuracy'
        
    Side Effects
    ------------
    Creates model performance charts showing:
    - Prediction accuracy over time
    - F1 score evolution
    - Model stability metrics
    - Feature importance trends (if available)
    """
```

### Analysis Functions

#### `create_summary_table()`

```python
def create_summary_table(data: Dict[str, pd.DataFrame]) -> None:
    """
    Create and display comprehensive performance summary table.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data
        
    Side Effects
    ------------
    Prints formatted table containing:
    - Portfolio vs benchmark returns
    - Risk metrics (volatility, tracking error, max drawdown)
    - Performance ratios (Sharpe, Information, Calmar)
    - Portfolio characteristics (holdings, concentration, turnover)
    - Cost analysis (transaction costs, fees)
    """
```

#### `run_complete_analysis()`

```python
def run_complete_analysis(data: Dict[str, pd.DataFrame]) -> None:
    """
    Run complete analysis suite with all visualizations and tables.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data from load_all_data()
        
    Side Effects
    ------------
    Executes full analysis workflow:
    1. Performance comparison charts
    2. Portfolio characteristics analysis
    3. Weight allocation visualization
    4. ML model performance evaluation
    5. Summary statistics table
    6. Business conclusions
    """
```

### Reporting Functions

#### `generate_markdown_executive_report()`

```python
def generate_markdown_executive_report(
    data: Dict[str, pd.DataFrame],
    summary_metrics: Dict,
    filename: str = 'executive_analysis_report.md'
) -> str:
    """
    Generate professional markdown executive report.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data
    summary_metrics : Dict
        Summary metrics from generate_executive_summary()
    filename : str, default 'executive_analysis_report.md'
        Output filename for the report
        
    Returns
    -------
    markdown_content : str
        Complete markdown report content
        
    Side Effects
    ------------
    Creates professional markdown report with:
    - Executive summary with key achievements
    - Performance highlights table
    - Portfolio characteristics analysis
    - Business impact assessment
    - Technical specifications
    - Methodology overview
    """
```

#### `export_summary_to_excel()`

```python
def export_summary_to_excel(
    data: Dict[str, pd.DataFrame],
    filename: str = 'mtum_analysis_complete.xlsx'
) -> None:
    """
    Export all analysis data to comprehensive Excel file.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data
    filename : str, default 'mtum_analysis_complete.xlsx'
        Output filename for Excel export
        
    Side Effects
    ------------
    Creates Excel file with separate sheets for:
    - Benchmark comparison metrics
    - Portfolio composition history
    - Portfolio weights evolution
    - Transaction costs analysis
    - Prediction accuracy metrics
    - Daily returns data
    """
```

#### `print_business_conclusions()`

```python
def print_business_conclusions(
    data: Dict[str, pd.DataFrame],
    summary_metrics: Dict
) -> None:
    """
    Print business impact and interview-ready conclusions.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data
    summary_metrics : Dict
        Summary metrics from generate_executive_summary()
        
    Side Effects
    ------------
    Prints structured business conclusions including:
    - Key achievements and operational advantages
    - Cost reduction analysis
    - Risk management benefits
    - Scalability opportunities
    - Technical excellence demonstrations
    - Recommended applications
    """
```

### Utility Functions

#### `quick_summary()`

```python
def quick_summary(data: Dict[str, pd.DataFrame]) -> None:
    """
    Print quick overview of loaded datasets.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded backtest data
        
    Side Effects
    ------------
    Prints dataset names and shapes for quick reference
    """
```

#### `reload_data()`

```python
def reload_data() -> Dict[str, pd.DataFrame]:
    """
    Reload data if files have changed.
    
    Returns
    -------
    data : Dict[str, pd.DataFrame]
        Refreshed data from latest backtest results
    """
```

#### `print_module_info()`

```python
def print_module_info() -> None:
    """
    Print comprehensive module information and usage guide.
    
    Side Effects
    ------------
    Displays detailed information about:
    - Available functions by category
    - Usage examples
    - Typical workflow patterns
    - Module version and description
    """
```

---

## Utility Functions

### Data Processing

#### `clean_data_pipeline_step()`

```python
def clean_data_pipeline_step(
    stage_1_portfolio_data: pd.DataFrame,
    stage_1_benchmark_data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete data cleaning step for monthly rebalancing loop.
    
    Parameters
    ----------
    stage_1_portfolio_data : pd.DataFrame
        Raw portfolio data requiring cleaning
    stage_1_benchmark_data : pd.DataFrame
        Raw benchmark data requiring cleaning
    start_date : pd.Timestamp
        Analysis start date for filtering
    end_date : pd.Timestamp
        Analysis end date for filtering
        
    Returns
    -------
    cleaned_portfolio_data : pd.DataFrame
        Cleaned and validated portfolio data
    cleaned_benchmark_data : pd.DataFrame
        Cleaned and validated benchmark data
    coverage_report : pd.DataFrame
        Data coverage statistics and quality metrics
        
    Examples
    --------
    >>> portfolio_clean, benchmark_clean, report = clean_data_pipeline_step(
    ...     stage_1_portfolio_data=raw_portfolio_data,
    ...     stage_1_benchmark_data=raw_benchmark_data,
    ...     start_date=pd.Timestamp('2023-01-01'),
    ...     end_date=pd.Timestamp('2024-01-01')
    ... )
    >>> print(f"Data coverage: {report['coverage_pct'].mean():.1%}")
    """
```

#### `prepare_returns_matrix_safe()`

```python
def prepare_returns_matrix_safe(
    df: pd.DataFrame,
    date_col: str = 'Date',
    ticker_col: str = 'Ticker',
    return_col: str = 'return'
) -> Optional[pd.DataFrame]:
    """
    Convert long format DataFrame to wide format returns matrix with error handling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format data with date, ticker, and return columns
    date_col : str, default 'Date'
        Date column name for index
    ticker_col : str, default 'Ticker'
        Ticker column name for columns
    return_col : str, default 'return'
        Return column name for values
        
    Returns
    -------
    returns_matrix : Optional[pd.DataFrame]
        Wide format returns matrix (dates x securities), None if error occurs
        
    Examples
    --------
    >>> returns_matrix = prepare_returns_matrix_safe(
    ...     df=portfolio_data,
    ...     date_col='Date',
    ...     ticker_col='Ticker',
    ...     return_col='daily_return'
    ... )
    >>> print(f"Matrix shape: {returns_matrix.shape}")
    >>> print(f"Date range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
    """
```

### File Organization

#### `organize_backtest_outputs()`

```python
def organize_backtest_outputs(
    results_prefix: str,
    create_subdirs: bool = True
) -> Tuple[str, Dict]:
    """
    Organize backtest output files into structured directories.
    
    Parameters
    ----------
    results_prefix : str
        Prefix of result files to organize (e.g., 'monthly_backtest_results_20241201_143022')
    create_subdirs : bool, default True
        Whether to create subdirectories (csv_files/, pickle_files/, excel_files/, charts/)
        
    Returns
    -------
    results_directory : str
        Path to organized results directory
    file_summary : Dict
        Summary of organized files by category:
        - 'csv_files': List of CSV files moved
        - 'pickle_files': List of pickle files moved
        - 'excel_files': List of Excel files moved
        - 'chart_files': List of chart files moved
        - 'other_files': List of other files moved
        
    Examples
    --------
    >>> results_dir, summary = organize_backtest_outputs(
    ...     results_prefix='monthly_backtest_results_20241201_143022',
    ...     create_subdirs=True
    ... )
    >>> print(f"Results organized in: {results_dir}")
    >>> print(f"CSV files: {len(summary['csv_files'])}")
    >>> print(f"Chart files: {len(summary['chart_files'])}")
    """
```

---

## Error Handling

### Exception Classes

```python
class PortfolioOptimizationError(Exception):
    """Raised when portfolio optimization fails."""
    
    def __init__(self, message: str, solver_status: str = None):
        self.solver_status = solver_status
        super().__init__(message)

class InsufficientDataError(Exception):
    """Raised when insufficient data for analysis."""
    
    def __init__(self, message: str, required_periods: int = None, available_periods: int = None):
        self.required_periods = required_periods
        self.available_periods = available_periods
        super().__init__(message)

class ModelTrainingError(Exception):
    """Raised when ML model training fails."""
    
    def __init__(self, message: str, model_type: str = None):
        self.model_type = model_type
        super().__init__(message)

class InvalidParameterError(Exception):
    """Raised when invalid parameters provided."""
    
    def __init__(self, message: str, parameter_name: str = None, valid_range: str = None):
        self.parameter_name = parameter_name
        self.valid_range = valid_range
        super().__init__(message)

class DataValidationError(Exception):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, failed_checks: List[str] = None):
        self.failed_checks = failed_checks or []
        super().__init__(message)
```

### Validation Functions

#### `validate_input_data()`

```python
def validate_input_data(
    df: pd.DataFrame,
    required_columns: List[str],
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    min_rows: int = 100
) -> Dict:
    """
    Validate input data for processing with comprehensive checks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data to validate
    required_columns : List[str]
        List of required column names
    date_range : Optional[Tuple[pd.Timestamp, pd.Timestamp]], default None
        Expected date range (start_date, end_date)
    min_rows : int, default 100
        Minimum required number of rows
        
    Returns
    -------
    validation_results : Dict
        Validation results containing:
        - 'is_valid': Overall validation status (bool)
        - 'missing_columns': List of missing required columns
        - 'data_coverage': Percentage of non-null values
        - 'date_range_coverage': Actual vs expected date coverage
        - 'row_count': Number of rows in dataset
        - 'warnings': List of validation warnings
        - 'errors': List of validation errors
        
    Raises
    ------
    DataValidationError
        If critical validation checks fail
        
    Examples
    --------
    >>> validation = validate_input_data(
    ...     df=portfolio_data,
    ...     required_columns=['Date', 'Ticker', 'Price'],
    ...     date_range=(pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01')),
    ...     min_rows=252
    ... )
    >>> if validation['is_valid']:
    ...     print("Data validation passed")
    ... else:
    ...     print(f"Validation errors: {validation['errors']}")
    """
```

#### `validate_optimization_inputs()`

```python
def validate_optimization_inputs(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    constraints: Dict
) -> Dict:
    """
    Validate inputs for portfolio optimization.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns vector
    covariance_matrix : pd.DataFrame
        Covariance matrix
    constraints : Dict
        Optimization constraints
        
    Returns
    -------
    validation_results : Dict
        Validation results for optimization inputs
        
    Raises
    ------
    InvalidParameterError
        If optimization inputs are invalid
    """
```

---

## Configuration Reference

### Default Parameters

```python
# XGBoost Model Configuration
DEFAULT_XGBOOST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5],
    'random_state': 42,
    'n_jobs': -1
}

# Portfolio Optimization Configuration
DEFAULT_OPTIMIZATION_PARAMS = {
    'risk_aversion': 1.0,
    'l1_penalty': 0.01,
    'max_weight': 0.30,
    'min_weight': 0.005,
    'max_positions': 20,
    'solver_preferences': ['ECOS', 'SCS', 'OSQP'],
    'max_solver_time': 60,
    'numerical_tolerance': 1e-6
}

# Feature Engineering Configuration
DEFAULT_FEATURE_COLUMNS = [
    'mom_change_1m_lag',    # 1-month momentum (lagged)
    'mom_change_3m_lag',    # 3-month momentum (lagged)
    'mom_change_6m_lag',    # 6-month momentum (lagged)
    'mom_change_12m_lag',   # 12-month momentum (lagged)
    'rolling_std_20d',      # 20-day rolling volatility
    'rolling_std_60d',      # 60-day rolling volatility
    'annualized_rolling_std_20d',  # Annualized 20-day volatility
    'annualized_rolling_std_60d',  # Annualized 60-day volatility
    'rolling_60d_corr'      # 60-day correlation with MTUM
]

# Backtesting Configuration
DEFAULT_BACKTEST_CONFIG = {
    'BULK_MODE': True,
    'SAVE_MONTHLY_CHARTS': False,
    'SAVE_EXPECTED_RETURNS_PLOTS': False,
    'SAVE_FEATURE_IMPORTANCE': False,
    'CHECKPOINT_EVERY_N_MONTHS': 24,
    'MINIMAL_LOGGING': True,
    'TRANSACTION_COST_BPS': 5.0,
    'RISK_FREE_RATE': 0.02,
    'BENCHMARK_TICKER': 'MTUM'
}

# Analysis Toolkit Configuration
DEFAULT_ANALYSIS_CONFIG = {
    'FIGURE_SIZE': (12, 8),
    'FONT_SIZE': 10,
    'CHART_STYLE': 'seaborn-v0_8',
    'SAVE_CHARTS': True,
    'CHART_DPI': 300,
    'CHART_FORMAT': 'png'
}
```

### Environment Variables

```python
# Optional environment variables for configuration
import os

# Data and file paths
DATA_PATH = os.getenv('MTUM_DATA_PATH', './raw_data.parquet')
RESULTS_PATH = os.getenv('MTUM_RESULTS_PATH', './results/')
CHARTS_PATH = os.getenv('MTUM_CHARTS_PATH', './charts/')

# Optimization parameters
RISK_AVERSION = float(os.getenv('MTUM_RISK_AVERSION', '1.0'))
L1_PENALTY = float(os.getenv('MTUM_L1_PENALTY', '0.01'))
MAX_POSITIONS = int(os.getenv('MTUM_MAX_POSITIONS', '20'))

# Processing configuration
MAX_WORKERS = int(os.getenv('MTUM_MAX_WORKERS', '-1'))
MEMORY_LIMIT_GB = int(os.getenv('MTUM_MEMORY_LIMIT', '8'))
SOLVER_TIME_LIMIT = int(os.getenv('MTUM_SOLVER_TIME_LIMIT', '60'))

# Analysis configuration
ENABLE_CHARTS = os.getenv('MTUM_ENABLE_CHARTS', 'true').lower() == 'true'
CHART_FORMAT = os.getenv('MTUM_CHART_FORMAT', 'png')
REPORT_FORMAT = os.getenv('MTUM_REPORT_FORMAT', 'markdown')
```

### Advanced Configuration

#### Custom Solver Configuration

```python
# CVXPY solver configuration for different scenarios
SOLVER_CONFIGS = {
    'fast': {
        'solvers': ['ECOS'],
        'max_iters': 1000,
        'tolerance': 1e-4
    },
    'accurate': {
        'solvers': ['ECOS', 'SCS'],
        'max_iters': 10000,
        'tolerance': 1e-8
    },
    'robust': {
        'solvers': ['ECOS', 'SCS', 'OSQP'],
        'max_iters': 5000,
        'tolerance': 1e-6,
        'retry_failed': True
    }
}
```

#### Feature Engineering Presets

```python
# Predefined feature sets for different strategies
FEATURE_PRESETS = {
    'momentum_focused': [
        'mom_change_1m_lag', 'mom_change_3m_lag', 
        'mom_change_6m_lag', 'mom_change_12m_lag'
    ],
    'risk_adjusted': [
        'mom_change_6m_lag', 'mom_change_12m_lag',
        'rolling_std_20d', 'rolling_std_60d',
        'rolling_60d_corr'
    ],
    'comprehensive': DEFAULT_FEATURE_COLUMNS,
    'minimal': [
        'mom_change_6m_lag', 'rolling_std_60d', 'rolling_60d_corr'
    ]
}
```

#### Performance Thresholds

```python
# Quality thresholds for automated validation
PERFORMANCE_THRESHOLDS = {
    'minimum_auc': 0.55,
    'target_auc': 0.70,
    'maximum_tracking_error': 0.15,
    'target_tracking_error': 0.05,
    'minimum_information_ratio': 0.0,
    'target_information_ratio': 0.5,
    'maximum_turnover_monthly': 1.0,
    'target_turnover_monthly': 0.3
}
```

---

## Usage Examples

### Complete Workflow Example

```python
# Stage 1: Portfolio Construction
import pandas as pd
from xgb_portfolio_model import train_xgb_portfolio_model
from portfolio_optimization import optimize_portfolio_cvxpy
from backtest_storage import BacktestResultsStorage

# Load and prepare data
raw_data = pd.read_parquet('raw_data.parquet')

# Train ML model
model, importance, data, y_test, y_prob, results = train_xgb_portfolio_model(
    df=historical_features_df,
    feature_columns=DEFAULT_FEATURE_COLUMNS,
    target_col='include_next_month'
)

# Optimize portfolio
weights, opt_results = optimize_portfolio_cvxpy(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    risk_aversion=1.0,
    l1_penalty=0.01
)

# Store results
storage = BacktestResultsStorage(
    start_date='2023-01-01',
    end_date='2024-01-01'
)
storage.store_monthly_results(
    rebalance_date='2023-01-31',
    portfolio_weights=weights_df,
    optimization_metrics=opt_results,
    benchmark_metrics=benchmark_comparison
)

# Stage 2: Professional Analysis
import mtum_analysis_toolkit as mtum

# Load and analyze results
data = mtum.load_all_data()
summary_metrics = mtum.generate_executive_summary(data)

# Generate visualizations
mtum.plot_cumulative_performance(data)
mtum.plot_portfolio_characteristics(data)
mtum.plot_portfolio_weights(data)

# Create professional reports
mtum.generate_markdown_executive_report(data, summary_metrics)
mtum.export_summary_to_excel(data)
mtum.print_business_conclusions(data, summary_metrics)
```

This comprehensive API documentation provides complete reference information for all functions, classes, and utilities in the MTUM Replica Portfolio System, enabling developers to effectively use and extend the system for systematic portfolio management and professional analysis.