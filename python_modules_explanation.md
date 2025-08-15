# Portfolio Optimization System - Python Modules Overview

## Three-Module Portfolio Management Pipeline

This system implements a complete quantitative portfolio management workflow using machine learning for security selection, mathematical optimization for portfolio construction, and comprehensive performance tracking.

---

## 1. xgb_portfolio_model.py
**Machine Learning Security Selection Engine**

**Purpose:** Uses XGBoost machine learning to predict which securities should be included in next month's portfolio based on historical momentum and volatility patterns.

**What it does:**
- **Feature Engineering**: Processes momentum indicators (1m, 3m, 6m, 12m returns), volatility metrics (20d, 60d rolling std), and correlation features
- **Binary Classification**: Trains XGBoost model to predict "include/exclude" decisions for each security next month
- **Target Creation**: Uses tracking error thresholds to determine which securities historically performed well vs benchmark
- **Prediction Pipeline**: Generates forward-looking predictions for portfolio inclusion decisions
- **Model Validation**: Provides accuracy metrics, feature importance, and performance visualization

**Key Functions:**
- `train_xgb_portfolio_model()` - Trains the classifier on historical data
- `predict_next_month_portfolio()` - Generates predictions for next month's securities
- `plot_model_performance()` - Visualizes model accuracy and feature importance

**Input:** Historical price/return data with calculated momentum and volatility features
**Output:** List of securities to include in next month's portfolio with confidence scores

---

## 2. portfolio_optimization.py  
**Mathematical Portfolio Construction Engine**

**Purpose:** Uses CVXPY convex optimization to construct optimal portfolio weights from the securities selected by the machine learning model.

**What it does:**
- **Mean-Variance Optimization**: Implements modern portfolio theory to balance expected returns vs risk
- **L1 Regularization**: Adds sparsity penalty to create focused portfolios with fewer positions
- **Risk Management**: Enforces position size limits, diversification constraints, and volatility targets
- **Benchmark Comparison**: Calculates comprehensive performance metrics vs benchmark (Sharpe ratio, alpha, beta, tracking error)
- **Robustness Testing**: Compares different optimization parameters to find optimal sparsity levels

**Key Functions:**
- `prepare_cvxpy_data()` - Prepares return matrices and covariance estimates
- `optimize_portfolio_cvxpy()` - Solves the constrained optimization problem
- `calculate_benchmark_comparison_metrics()` - Comprehensive performance analysis
- `compare_regularization_levels()` - Tests different L1 penalty parameters

**Input:** Selected securities from XGBoost + historical returns data
**Output:** Optimal portfolio weights, expected returns/risk, and performance metrics

---

## 3. backtest_storage.py
**Comprehensive Results Storage & Analysis System**

**Purpose:** Captures, stores, and analyzes all results from the monthly backtesting process for later performance evaluation and research.

**What it does:**
- **Monthly Storage**: Records portfolio weights, optimization parameters, prediction accuracy, and performance metrics for each rebalancing period
- **Daily Performance Tracking**: Stores daily portfolio returns, benchmark returns, and position values
- **Transaction Cost Analysis**: Calculates portfolio turnover, trading costs, and rebalancing impact
- **Benchmark Comparison**: Tracks alpha, beta, information ratio, drawdowns, and capture ratios over time
- **Results Persistence**: Saves all data to CSV/Excel files and provides summary statistics

**Key Functions:**
- `BacktestResultsStorage()` - Main storage class for organizing all backtest data
- `store_monthly_results()` - Records all metrics from each month's rebalancing
- `calculate_backtest_summary_stats()` - Computes overall performance statistics
- `get_summary_dataframes()` - Converts stored data to pandas DataFrames for analysis

**Input:** Results from XGBoost predictions and CVXPY optimization each month
**Output:** Comprehensive database of backtest results, performance reports, and analysis-ready datasets

---

## How They Work Together

```
1. Historical Data → xgb_portfolio_model.py → Securities to Include Next Month
                                ↓
2. Selected Securities → portfolio_optimization.py → Optimal Portfolio Weights  
                                ↓
3. All Results → backtest_storage.py → Performance Database & Analysis
```

**Monthly Workflow:**
1. **XGBoost** analyzes 13 months of data and predicts which securities to include
2. **CVXPY** optimizes portfolio weights for the selected securities  
3. **Storage System** records all results and calculates performance metrics
4. Process repeats for next month with updated data

**Final Output:**
- Complete backtest database with 2+ years of monthly results
- Performance attribution and risk analysis
- Transaction cost and turnover analysis  
- Benchmark comparison and alpha generation analysis
- Ready-to-analyze datasets for further research

This modular design separates concerns (ML prediction, optimization, storage) while maintaining a clean data flow for systematic portfolio management and research.