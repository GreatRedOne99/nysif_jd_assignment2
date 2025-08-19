# MTUM Replica Portfolio System

## Overview

This project implements a systematic portfolio management system that replicates the **MTUM (iShares MSCI USA Momentum Factor ETF)** using machine learning and mathematical optimization. The system creates a sparse, cost-efficient portfolio that tracks MTUM performance while using significantly fewer holdings.

### What This System Does

🧠 **Machine Learning**: XGBoost predicts which securities to include based on momentum and volatility patterns  
🎯 **Portfolio Optimization**: CVXPY optimizes weights using mean-variance optimization with L1 regularization  
📊 **Performance Tracking**: Comprehensive backtesting with institutional-quality performance metrics  
📈 **Professional Analysis**: Automated report generation with executive summaries and visualizations  

### Results
Achieves 2-4% tracking error while reducing holdings from 200+ to 15-25 securities with comprehensive analysis reporting.

### The Complete Pipeline
```
Stage 1: Portfolio Construction
Raw Data → Feature Engineering → ML Predictions → Portfolio Optimization → Backtest Results

Stage 2: Professional Analysis  
Backtest Results → Automated Analysis → Executive Reports → Investment Presentations
```

## Quick Start Guide

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB free space
- **Git**: (Optional) For cloning repository

### Installation Steps

#### 1. Get the Project Files

Choose **Option A** (Google CoLab - Recommended) or **Option B** (Git Clone):


**Option A: Google CoLab**
- Open the primary notebook - nysif_jd_assignment2.ipynb - in Google CoLab<br>

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GreatRedOne99/nysif_jd_assignment2/blob/main/jd_assignment2.ipynb)

**Option B: Clone from GitHub (Recommended)**
```bash
# Clone the repository
git clone https://github.com/GreatRedOne99/nysif_jd_assignment2.git

# Navigate to project directory
cd nysif_jd_assignment2

# Verify all files are present
ls -la  # Linux/macOS
dir     # Windows
```

#### 2. Get Market Data

**Option A: Fresh Data (Recommended)**
1. Upload `Google_CoLab_yfinance_downloader.ipynb` to Google Colab
2. Run all cells in Colab
3. Download the generated `raw_data.parquet` file
4. Save to your project directory

**Option B: Use Included Data**
- Use the provided `raw_data.parquet` file (backup option)

#### 3. Setup Python Environment

Open command prompt/terminal in project directory and run:

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Setup Jupyter kernel
python -m ipykernel install --user --name=nysif_jd_assign2 --display-name="nysif-jd-assign2"

# Start Jupyter Notebook
jupyter notebook
```

**Git Users: Keep Project Updated**
```bash
# Pull latest changes
git pull origin main

# Check for updates
git status
```

## Complete Workflow: Two-Stage Pipeline

### Stage 1: Portfolio Construction & Backtesting

#### 4. Run the Main Analysis
1. Open `jd_assignment2.ipynb` in Jupyter
2. Click 'Trust' notebook if prompted
3. Select the "nysif-jd-assign2" kernel
4. **Important**: Run cells in sequence - do not use "Run All"
5. **Configure Analysis Period** (Cell ~8-9):
   - From Cell 8: Review the Month End Dates array
   - In Cell 9: Set these variables to control analysis scope:
     ```python
     START_INDEX = -6    # Start from 6th month from end
     END_INDEX = -1      # End at last month
     ```
   - **Recommendation**: Start with 5-6 months for faster execution
   - For full analysis (24+ months), allow 2-3 hours runtime
6. Continue running cells sequentially until completion
7. Results will be saved in timestamped folders like `backtest_results_YYYYMMDD_HHMMSS/`

### Stage 2: Professional Analysis & Reporting

#### 5. Run the Analysis Notebook
After Stage 1 completes:

1. **Save the Analysis Toolkit**
   - Copy `mtum_analysis_toolkit.py` to your project directory
   - This contains all analysis functions

2. **Open Analysis Notebook**
   - Create new notebook: `mtum_analysis.ipynb`
   - Or copy the provided clean analysis template

3. **Run Complete Analysis**
   ```python
   # Cell 1: Setup
   import mtum_analysis_toolkit as mtum
   mtum.print_module_info()

   # Cell 2: Auto-load Latest Results  
   data = mtum.load_all_data()
   summary_metrics = mtum.generate_executive_summary(data)

   # Cell 3: Analysis with dates
   summary = mtum.generate_executive_summary(data)

   # Cell 4: Reports with dates
   mtum.generate_markdown_executive_report(data, summary)
   mtum.export_summary_to_excel(data)

   # Cell 5: Complete analysis
   mtum.run_complete_analysis(data)
   mtum.print_business_conclusions(data, summary)

   # Cell 6: Generate Professional Reports
   mtum.generate_markdown_executive_report(data, summary_metrics)
   mtum.export_summary_to_excel(data)
   ```

## Project Files

### Core Analysis Files
| File | Purpose |
|------|---------|
| **jd_assignment2.ipynb** | 🎯 **Stage 1: Main backtest notebook - START HERE** |
| **mtum_analysis_toolkit.py** | 🔧 **Stage 2: Analysis functions module** |
| **mtum_analysis.ipynb** | 📊 **Stage 2: Clean analysis notebook template** |
| **requirements.txt** | Python dependencies |
| **raw_data.parquet** | Market data (ETF prices) |

### Support Modules (Stage 1)
| File | Purpose |
|------|---------|
| **xgb_portfolio_model.py** | XGBoost ML model for security selection |
| **portfolio_optimization.py** | CVXPY portfolio optimization with L1 regularization |
| **backtest_storage.py** | Performance tracking and results storage |
| **expected_returns_models.py** | Ex-ante expected returns calculation |
| **portfolio_performance_summary.py** | Performance analysis and visualization |
| **portfolio_precleaning.py** | Data cleaning and preprocessing |

### Documentation & Reference  
| File | Purpose |
|------|---------|
| **README.md** | 📖 This overview and quick start guide |
| **user_guide.md** | 👥 Detailed user instructions |
| **technical_reference.md** | 🔬 Technical implementation details |
| **api_documentation.md** | 🔌 Complete API reference |
| **reference.md** | 📚 MTUM methodology and background |
| **problem_statement.ipynb** | Original assignment requirements |
| **Google_CoLab_yfinance_downloader.ipynb** | Data download utility |

## Expected Results

### Stage 1 Output: Backtest Results
After `jd_assignment2.ipynb` completes, you'll get:

```
backtest_results_YYYYMMDD_HHMMSS/
├── csv_files/              # Performance metrics tables
├── charts/                 # Individual performance charts
├── excel_files/            # Excel summary reports
├── pickle_files/           # Portfolio weights data
└── other_files/            # Configuration and metadata
```

### Stage 2 Output: Professional Analysis
After `mtum_analysis.ipynb` completes, you'll get:

📄 **Executive Reports**
- `executive_analysis_report.md` - Professional markdown report with dates
- `mtum_analysis_complete.xlsx` - All data in Excel format

📊 **Analysis Charts**
- Cumulative returns comparison (Portfolio vs MTUM)
- Portfolio characteristics evolution
- Current portfolio weights allocation
- ML model performance metrics
- Transaction cost analysis

📋 **Key Performance Metrics**
- **Tracking Error**: ~3% (target: 2-4%)
- **Information Ratio**: >0.5 (good active management)
- **Portfolio Size**: 15-25 holdings (vs 200+ in MTUM)
- **Annual Turnover**: ~150% (reasonable for momentum strategy)

### Analysis Presentations & Business Intelligence

After `mtum_analysis.ipynb` completes, you'll also get **presentation-ready materials**:

📊 **Interactive Analysis Components**
- Executive summary with key performance metrics
- Business impact conclusions and recommendations
- Professional visualizations suitable for client presentations
- Automated period detection with precise analysis dates

🎯 **Interview-Ready Talking Points**
- Key achievements and operational advantages
- Technical excellence demonstrations
- Recommended applications for other portfolios
- Risk management and cost reduction insights

💼 **Business Conclusions Available**
- Portfolio efficiency analysis (15-25 holdings vs 200+ in MTUM)
- Information ratio performance assessment
- Transaction cost optimization results
- Alpha generation and beta exposure analysis

📈 **Generated Analysis Presentations Include**
- Cumulative performance comparison charts
- Portfolio characteristics evolution over time
- Current portfolio weights allocation visualization
- ML model performance and accuracy metrics
- Transaction cost analysis and turnover metrics

## System Architecture

### Stage 1: Portfolio Construction Pipeline
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Market    │    │   Feature       │    │   Machine       │
│   Data Input    │───▶│   Engineering   │───▶│   Learning      │
│                 │    │                 │    │   Prediction    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Backtest      │    │   Portfolio     │    │   Portfolio     │
│   Results       │◀────│   Performance   │◀────│   Optimization  │
│   Storage       │    │   Tracking      │    │   (CVXPY)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Stage 2: Professional Analysis Pipeline
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Backtest      │    │   Automated     │    │   Executive     │
│   Results       │───▶│   Analysis      │───▶│   Reports &     │
│   Files         │    │   Generation    │    │   Presentations │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Business      │
                       │   Intelligence  │
                       │   & Metrics     │
                       │   Exploration   │
                       └─────────────────┘
```

## Complete Analysis Workflow - Updated

```python
# Stage 2: Professional Analysis & Reporting (Updated)
# Cell 1: Setup and Module Info
import mtum_analysis_toolkit as mtum
mtum.print_module_info()

# Cell 2: Auto-load Latest Results  
data = mtum.load_all_data()
summary_metrics = mtum.generate_executive_summary(data)

# Cell 3-6: Core Visualizations
mtum.plot_cumulative_performance(data)           # Performance comparison
mtum.plot_portfolio_characteristics(data)        # Portfolio evolution
mtum.plot_portfolio_weights(data)                # Current allocations
mtum.plot_model_performance(data)                # ML accuracy metrics

# Cell 7-8: Business Analysis
mtum.create_summary_table(data)                  # Comprehensive metrics table
mtum.print_business_conclusions(data, summary_metrics)  # Interview talking points

# Cell 9-10: Professional Reports Generation
mtum.generate_markdown_executive_report(data, summary_metrics)  # Executive report
mtum.export_summary_to_excel(data)              # Excel export for further analysis
```

## Key Features

### Stage 1: Portfolio Construction
- **End-to-end systematic process** from data to portfolio construction
- **Machine learning integration** with traditional optimization methods
- **Production-ready implementation** with proper validation and risk controls
- **Comprehensive backtesting** with monthly rebalancing

### Stage 2: Professional Analysis
- **Automatic file detection** - finds latest backtest results automatically
- **Comprehensive visualizations** - institutional-quality charts and analysis
- **Executive reporting** - markdown reports with complete analysis period dates
- **Business conclusions** - interview-ready talking points and insights
- **Modular design** - reusable analysis toolkit for future backtests

## Troubleshooting

### Stage 1 Issues

**"Insufficient tickers" Error**
- Check that `raw_data.parquet` is in the project directory
- Ensure data covers the full analysis period

**Memory Issues**
- Reduce analysis period using START_INDEX/END_INDEX variables
- Enable bulk mode processing in notebook
- Close other applications to free RAM

**Optimization Failures**
- Check that selected securities have sufficient data
- Verify covariance matrix is positive definite
- Relax constraints if optimization fails

### Stage 2 Issues

**"Module not found" Error**
```
ModuleNotFoundError: No module named 'mtum_analysis_toolkit'
```
**Solution**: Ensure `mtum_analysis_toolkit.py` is saved in the same directory as your notebook

**"No backtest results found" Error**
```
⚠ No backtest results directories found!
```
**Solution**: 
- Run Stage 1 (`jd_assignment2.ipynb`) first to generate results
- Ensure results directory exists with pattern `backtest_results_*`

**Charts not displaying**
**Solution**: Add `plt.show()` after plot functions if running in Jupyter

### Git-Specific Issues

**Git Clone Permission Issues**
```bash
# If you get permission errors, try:
git clone https://github.com/GreatRedOne99/nysif_jd_assignment2.git --depth 1
```

**Keeping Project Updated**
```bash
# Check for updates
git status
git pull origin main

# If you have local changes, stash them first
git stash
git pull origin main
git stash pop
```

**Large File Issues**
- The `raw_data.parquet` file may not be in Git due to size
- Always run the data download step regardless of installation method

## Documentation Guide

### For Users
- **README.md** (this file) - Overview and quick start
- **user_guide.md** - Detailed step-by-step instructions
- **reference.md** - MTUM methodology and background

### For Developers  
- **technical_reference.md** - System architecture and mathematical framework
- **api_documentation.md** - Complete function and class reference

### For Analysts
- **Executive reports** - Auto-generated from Stage 2 analysis with presentation materials
- **Excel exports** - All data and metrics for custom presentations
- **Interactive notebooks** - Customizable analysis and presentation generation

## Academic Context

This system implements concepts from:
- **Modern Portfolio Theory** (Markowitz optimization)
- **Factor Investing** (Momentum factor replication)  
- **Machine Learning in Finance** (Predictive modeling)
- **Systematic Portfolio Management** (Quantitative strategies)

## Support

**Having Issues?**
- Check the troubleshooting section above
- Review the user guide for detailed instructions
- Verify all dependencies are installed correctly
- Ensure you have sufficient system resources

**For Development**:
- See `technical_reference.md` for system architecture
- Refer to `api_documentation.md` for function details
- Check individual module docstrings for API details

**Contributing via Git**:
- Fork the repository for your own modifications
- Create feature branches for significant changes
- Submit pull requests for improvements

---

**🚀 Ready to get started?** 

**Option A: Git Clone (Recommended)**
```bash
git clone https://github.com/GreatRedOne99/nysif_jd_assignment2.git
cd nysif_jd_assignment2
```

**Option B: ZIP Download**
- Download and extract the ZIP file
- Navigate to project directory

**Then:**
1. **First**: Run `jd_assignment2.ipynb` (Stage 1) to generate backtest results
2. **Then**: Run `mtum_analysis.ipynb` (Stage 2) for professional analysis and reporting

This gives you both the systematic portfolio and professional analysis tools for interviews and presentations!
