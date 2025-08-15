# User Guide - MTUM Replica Portfolio System

## Table of Contents
1. [Getting Started](#getting-started)
2. [System Architecture](#system-architecture)
3. [Complete Two-Stage Workflow](#complete-two-stage-workflow)
4. [Configuration Options](#configuration-options)
5. [Understanding Results](#understanding-results)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (for full backtest)
- Internet connection (for data download)
- Basic understanding of portfolio management concepts

### Installation Steps

1. **Environment Setup**
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

2. **Jupyter Setup**
```bash
# Install Jupyter kernel
python -m ipykernel install --user --name=nysif_jd_assign2 --display-name="nysif-jd-assign2"

# Start Jupyter
jupyter notebook
```

3. **Data Preparation**
Choose one option:

**Option A: Fresh Data Download (Recommended)**
- Upload `Google_CoLab_yfinance_downloader.ipynb` to Google Colab
- Run all cells to download latest market data
- Save `raw_data.parquet` to your project directory

**Option B: Use Included Data**
- Use the provided `raw_data.parquet` file
- Data covers approximately 5+ years of historical prices

## System Architecture

### The Complete Two-Stage Pipeline

```
Stage 1: Portfolio Construction & Backtesting

Stage 1: Portfolio Construction & Backtesting
├── Feature Engineering & ML Prediction
│   ├── Calculate momentum factors (1M, 3M, 6M, 12M)
│   ├── Compute volatility metrics (20D, 60D rolling)
│   ├── Calculate correlations with MTUM
│   └── XGBoost classification for security selection
│
├── Portfolio Optimization
│   ├── Ex-ante expected returns calculation
│   ├── Risk model construction (covariance matrix)
│   ├── CVXPY optimization with constraints
│   └── Portfolio weight determination
│
└── Performance Backtesting
   ├── Daily return calculation
   ├── Benchmark comparison metrics
   ├── Transaction cost analysis
   └── Results storage and organization

Stage 2: Professional Analysis & Reporting

Stage 2: Professional Analysis & Reporting
├── Automated Data Loading
│   ├── Find latest backtest results automatically
│   ├── Load all performance data and metrics
│   └── Validate data completeness
│
├── Advanced Analytics
│   ├── Executive summary with analysis period
│   ├── Professional visualizations
│   ├── Portfolio characteristics analysis
│   └── ML model performance evaluation
│
└── Professional Reporting
   ├── Executive markdown reports
   ├── Excel data exports
   ├── Business intelligence summaries
   └── Interview-ready talking points
```

## Complete Two-Stage Workflow

### STAGE 1: Portfolio Construction & Backtesting

#### Phase 1: Data Loading and Preprocessing

1. **Open Main Notebook**
   - Launch `jd_assignment2.ipynb`
   - Ensure kernel is set to "nysif-jd-assign2"

2. **Load Raw Data**
```python
# The system automatically loads data
raw_data = pd.read_parquet('raw_data.parquet')
print(f"Loaded {len(raw_data)} rows of market data")
```

3. **Data Quality Checks**
   - Missing data handling
   - Corporate action adjustments
   - Survivorship bias correction

#### Phase 2: Feature Engineering

The system automatically calculates key features:

**Momentum Features**
- 1-month lagged momentum: `mom_change_1m_lag`
- 3-month lagged momentum: `mom_change_3m_lag`
- 6-month lagged momentum: `mom_change_6m_lag`
- 12-month lagged momentum: `mom_change_12m_lag`

**Risk Features**
- 20-day rolling volatility: `rolling_std_20d`
- 60-day rolling volatility: `rolling_std_60d`
- Annualized volatilities: `annualized_rolling_std_*`

**Market Relationship**
- 60-day correlation with MTUM: `rolling_60d_corr`

#### Phase 3: Machine Learning Model

**Training Process**
1. Time-series cross-validation to prevent look-ahead bias
2. Hyperparameter optimization using GridSearchCV
3. Feature importance analysis
4. Out-of-sample validation

**Key Metrics to Monitor**
- **AUC Score**: >0.6 indicates predictive power
- **Precision/Recall**: Balance between inclusion accuracy
- **Feature Importance**: Understanding model drivers

#### Phase 4: Portfolio Construction

**Expected Returns Calculation**
Choose from multiple methods:
- `historical`: Simple historical means
- `momentum`: Momentum-based forecasts
- `multi_factor`: Advanced multi-factor model (recommended)
- `black_litterman`: Bayesian approach

**Optimization Process**
```python
# Key parameters you can adjust
optimization_params = {
    'risk_aversion': 1.0,      # Higher = more conservative
    'l1_penalty': 0.01,        # Higher = more sparse portfolio
    'max_weight': 0.30,        # Maximum position size
    'min_weight': 0.005,       # Minimum meaningful position
    'max_positions': 20        # Portfolio size constraint
}
```

#### Phase 5: Backtesting

**Monthly Rebalancing Loop**
The system processes each month:
1. Update features with latest data
2. Generate ML predictions
3. Optimize portfolio weights
4. Calculate transaction costs
5. Store performance metrics

**Progress Monitoring**
- Real-time progress updates
- Success/failure tracking
- Performance metric display

#### Phase 6: Results Organization

After completion, Stage 1 creates:
```
backtest_results_YYYYMMDD_HHMMSS/
├── csv_files/              # Performance metrics tables
├── charts/                 # Individual performance charts
├── excel_files/            # Excel summary reports
├── pickle_files/           # Portfolio weights data
└── other_files/            # Configuration and metadata
```

---

### STAGE 2: Professional Analysis & Reporting

#### Phase 7: Analysis Setup

1. **Save Analysis Toolkit**
   - Ensure `mtum_analysis_toolkit.py` is in your project directory
   - This contains all professional analysis functions

2. **Create Analysis Notebook**
   - Create new notebook: `mtum_analysis.ipynb`
   - Or use provided analysis template

3. **Import and Setup**
```python
# Cell 1: Import and module information
import mtum_analysis_toolkit as mtum
mtum.print_module_info()
```

#### Phase 8: Data Loading and Summary

```python
# Cell 2: Auto-load latest backtest results
data = mtum.load_all_data()
summary_metrics = mtum.generate_executive_summary(data)
```

**What This Does:**
- 🔍 **Automatically finds** latest backtest results directory
- 📊 **Loads all data files**: benchmark comparison, portfolio weights, transaction costs, etc.
- 📋 **Generates executive summary** with precise analysis dates and key metrics
- ⚡ **Validates data completeness** and reports any issues

#### Phase 9: Professional Visualizations

```python
# Cell 3: Performance comparison
mtum.plot_cumulative_performance(data)
```
**Creates:** Professional cumulative returns chart comparing portfolio vs MTUM

```python
# Cell 4: Portfolio characteristics
mtum.plot_portfolio_characteristics(data)
```
**Creates:** Multi-panel chart showing portfolio evolution over time

```python
# Cell 5: Portfolio weights analysis
mtum.plot_portfolio_weights(data)
```
**Creates:** Current allocation pie chart and historical weight trends

```python
# Cell 6: ML model performance
mtum.plot_model_performance(data)
```
**Creates:** Model accuracy and prediction performance over time

#### Phase 10: Business Analysis

```python
# Cell 7: Comprehensive summary table
mtum.create_summary_table(data)
```
**Displays:** Formatted table with all key performance metrics

```python
# Cell 8: Business conclusions and talking points
mtum.print_business_conclusions(data, summary_metrics)
```
**Provides:** Interview-ready business conclusions and operational advantages

#### Phase 11: Professional Reports

```python
# Cell 9: Generate executive report
mtum.generate_markdown_executive_report(data, summary_metrics)
```
**Creates:** `executive_analysis_report.md` with complete professional analysis

```python
# Cell 10: Export to Excel
mtum.export_summary_to_excel(data)
```
**Creates:** `mtum_analysis_complete.xlsx` with all data for further analysis

#### Phase 12: Final Analysis Outputs

After Stage 2 completion, you'll have:

📄 **Executive Reports**
- `executive_analysis_report.md` - Professional markdown report with precise dates
- `mtum_analysis_complete.xlsx` - All data in Excel format

📊 **Presentation-Ready Materials**
- Cumulative returns comparison (Portfolio vs MTUM)
- Portfolio characteristics evolution over analysis period
- Current portfolio weights allocation breakdown
- ML model performance and prediction accuracy metrics
- Transaction cost analysis and portfolio turnover assessment

💼 **Business Intelligence**
- Key achievements and operational advantages
- Performance metrics with industry context
- Cost reduction analysis vs benchmark
- Risk management and diversification benefits
- Scaling opportunities and recommended applications

## Configuration Options

### Stage 1: Backtest Configuration

**Bulk Mode Settings (Recommended for Production)**
```python
# In jd_assignment2.ipynb
BULK_MODE = True
SAVE_MONTHLY_CHARTS = False
SAVE_EXPECTED_RETURNS_PLOTS = False
SAVE_FEATURE_IMPORTANCE = False
CHECKPOINT_EVERY_N_MONTHS = 24
MINIMAL_LOGGING = True
```

**Research Mode Settings (For Analysis)**
```python
# For detailed analysis and debugging
BULK_MODE = False
SAVE_MONTHLY_CHARTS = True
SAVE_EXPECTED_RETURNS_PLOTS = True
SAVE_FEATURE_IMPORTANCE = True
MINIMAL_LOGGING = False
```

**Date Range Configuration**
```python
# Configure analysis period (Cell ~8-9 in notebook)
START_INDEX = -6    # Start from 6th month from end
END_INDEX = -1      # End at last month

# For full historical analysis
START_INDEX = 0     # Start from first available month
END_INDEX = -1      # Process through last month
```

### Stage 2: Analysis Configuration

**Chart and Report Settings**
```python
# In mtum_analysis.ipynb - these are handled automatically
# But you can customize if needed:

# Chart settings
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

# Report settings
executive_report_filename = 'executive_analysis_report.md'
excel_export_filename = 'mtum_analysis_complete.xlsx'
```

## Understanding Results

### Stage 1: Backtest Results

**Key Performance Metrics**

**Tracking Error**
- Target: 2-4% annualized
- Lower = closer to benchmark tracking
- Higher = more active management approach

**Information Ratio**
- Target: >0.5 for good active management
- Measures excess return per unit of tracking error
- Higher = better risk-adjusted active performance

**Sharpe Ratio**
- Measures overall risk-adjusted returns
- Compare portfolio vs benchmark Sharpe ratios
- Higher = better risk-adjusted performance

**Portfolio Characteristics**
- **Holdings Count**: Typically 10-25 securities (vs 200+ in MTUM)
- **Concentration**: Top 5 holdings typically 40-70%
- **Turnover**: Monthly turnover typically 5-20%
- **Transaction Costs**: Typically 3-8 basis points per month

### Stage 2: Analysis Results

**Executive Summary Metrics**
- **Analysis Period**: Exact start and end dates with duration
- **Performance Comparison**: Portfolio vs benchmark returns
- **Risk Metrics**: Volatility, tracking error, maximum drawdown
- **Efficiency Metrics**: Holdings reduction, cost savings

**Business Intelligence**
- **Operational Advantages**: Reduced complexity, lower costs
- **Risk Management**: Systematic tracking error control
- **Scalability**: Replicable process for other factors
- **Technical Excellence**: ML + optimization integration

**Professional Reports**
- **Markdown Report**: Complete analysis with business context
- **Excel Export**: All data for custom analysis
- **Visualizations**: Publication-ready charts and graphs

## Troubleshooting

### Stage 1 Issues

**1. Insufficient Data Error**
```
Error: Insufficient tickers (2) for analysis
```
**Solution**: 
- Check that `raw_data.parquet` is in project directory
- Verify data covers the full analysis period
- Adjust `min_data_coverage` parameter if needed

**2. Memory Issues**
```
MemoryError: Unable to allocate array
```
**Solution**:
- Reduce analysis period using START_INDEX/END_INDEX
- Enable BULK_MODE = True
- Close other applications to free RAM
- Consider using a machine with more memory

**3. Optimization Failures**
```
CVXPY optimization failed with status: INFEASIBLE
```
**Solution**:
- Check that selected securities have sufficient data
- Relax L1 penalty (reduce l1_penalty parameter)
- Increase max_weight constraint
- Verify covariance matrix is positive definite

### Stage 2 Issues

**1. Module Not Found Error**
```
ModuleNotFoundError: No module named 'mtum_analysis_toolkit'
```
**Solution**: 
- Ensure `mtum_analysis_toolkit.py` is saved in the same directory as your notebook
- Check that the file is not corrupted or incomplete
- Restart the Jupyter kernel and try again

**2. No Backtest Results Found**
```
❌ No backtest results directories found!
```
**Solution**: 
- Run Stage 1 (`jd_assignment2.ipynb`) first to generate results
- Ensure results directory exists with pattern `backtest_results_*`
- Check that Stage 1 completed successfully

**3. Charts Not Displaying**
**Solution**: 
- Add `plt.show()` after plot functions if running in Jupyter
- Ensure matplotlib backend is properly configured
- Try restarting the Jupyter kernel

**4. Empty or Incomplete Data**
```
⚠️ Limited analysis available without complete data
```
**Solution**:
- Verify Stage 1 backtest completed fully
- Check for any errors in Stage 1 execution
- Ensure all CSV files were generated in results directory

## Best Practices

### Stage 1: Portfolio Construction

1. **Start Small**: Begin with 5-6 months for testing, then scale to full analysis
2. **Use Bulk Mode**: Enable for production runs to improve performance
3. **Monitor Progress**: Watch for optimization failures or data issues
4. **Save Checkpoints**: Use checkpoint feature for long backtests
5. **Validate Results**: Check that key metrics are reasonable

### Stage 2: Professional Analysis

1. **Always Run Stage 1 First**: Stage 2 requires Stage 1 backtest results
2. **Check Data Completeness**: Use `quick_summary()` to verify data loading
3. **Generate All Reports**: Create both markdown and Excel outputs
4. **Review Business Conclusions**: Use for interview preparation and presentations
5. **Customize Analysis**: Modify charts and reports as needed for specific audiences

### Performance Optimization

**For Long Backtests (24+ months)**:
- Use BULK_MODE = True
- Disable monthly chart generation
- Enable checkpointing every 12-24 months
- Monitor memory usage

**For Research and Development**:
- Use smaller date ranges for faster iteration
- Enable detailed logging and charts
- Save feature importance and model diagnostics
- Experiment with different parameters

### Data Management

1. **Backup Results**: Save important backtest results
2. **Organize Files**: Use the automatic directory organization
3. **Version Control**: Track parameter changes and results
4. **Document Changes**: Note any modifications to default settings

This comprehensive user guide provides step-by-step instructions for using both stages of the MTUM replica portfolio system effectively, from initial portfolio construction through professional analysis and reporting.