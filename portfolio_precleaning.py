import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def clean_portfolio_data_for_period(stage_1_portfolio_data, start_date, end_date, 
                                   min_data_coverage=0.50, verbose=True):
    """
    Clean portfolio data for a specific analysis period, handling survivorship bias
    and missing data issues.
    
    Parameters:
    stage_1_portfolio_data: DataFrame with Ticker, Date, Adj_Close columns
    start_date: Start date for analysis period
    end_date: End date for analysis period  
    min_data_coverage: Minimum fraction of trading days required (0.5 = 50%)
    verbose: Whether to print detailed logs
    
    Returns:
    Cleaned DataFrame ready for the rest of the pipeline
    """
    
    if verbose:
        print(f"🧹 Cleaning portfolio data for period: {start_date} to {end_date}")
    
    # 1. Filter by date range
    filtered_data = stage_1_portfolio_data[
        (stage_1_portfolio_data['Date'] >= start_date) & 
        (stage_1_portfolio_data['Date'] <= end_date)
    ].copy()
    
    if verbose:
        print(f"   After date filter: {len(filtered_data):,} rows, {filtered_data['Ticker'].nunique()} tickers")
    
    # 2. Remove rows with NaN prices immediately
    initial_nan_count = filtered_data['Adj_Close'].isna().sum()
    if initial_nan_count > 0:
        if verbose:
            nan_tickers = filtered_data[filtered_data['Adj_Close'].isna()]['Ticker'].unique()
            print(f"   Found {initial_nan_count} NaN prices in tickers: {list(nan_tickers)}")
        
        filtered_data = filtered_data.dropna(subset=['Adj_Close'])
        
        if verbose:
            print(f"   After removing NaN prices: {len(filtered_data):,} rows, {filtered_data['Ticker'].nunique()} tickers")
    
    # 3. Calculate data coverage per ticker
    total_trading_days = len(filtered_data['Date'].unique())
    min_required_days = int(total_trading_days * min_data_coverage)
    
    ticker_coverage = filtered_data.groupby('Ticker').agg({
        'Date': ['count', 'min', 'max'],
        'Adj_Close': ['count', 'mean']  # Count and average price as data quality check
    }).round(4)
    
    ticker_coverage.columns = ['obs_count', 'first_date', 'last_date', 'price_count', 'avg_price']
    ticker_coverage['coverage_pct'] = ticker_coverage['obs_count'] / total_trading_days
    ticker_coverage['days_in_period'] = (ticker_coverage['last_date'] - ticker_coverage['first_date']).dt.days
    
    # 4. Identify tickers to keep vs remove
    tickers_to_keep = ticker_coverage[
        ticker_coverage['obs_count'] >= min_required_days
    ].index.tolist()
    
    tickers_to_remove = ticker_coverage[
        ticker_coverage['obs_count'] < min_required_days
    ].index.tolist()
    
    if verbose:
        print(f"   Total trading days in period: {total_trading_days}")
        print(f"   Minimum required observations: {min_required_days} ({min_data_coverage:.0%})")
        
        if len(tickers_to_remove) > 0:
            print(f"\n   ❌ Removing {len(tickers_to_remove)} tickers with insufficient data:")
            for ticker in tickers_to_remove:
                obs = ticker_coverage.loc[ticker, 'obs_count']
                pct = ticker_coverage.loc[ticker, 'coverage_pct']
                first_date = ticker_coverage.loc[ticker, 'first_date']
                last_date = ticker_coverage.loc[ticker, 'last_date']
                print(f"      {ticker}: {obs}/{total_trading_days} days ({pct:.1%}) | {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
        
        print(f"\n   ✅ Keeping {len(tickers_to_keep)} tickers with sufficient data:")
        for ticker in tickers_to_keep:
            obs = ticker_coverage.loc[ticker, 'obs_count']
            pct = ticker_coverage.loc[ticker, 'coverage_pct']
            print(f"      {ticker}: {obs}/{total_trading_days} days ({pct:.1%})")
    
    # 5. Filter to keep only good tickers
    cleaned_data = filtered_data[filtered_data['Ticker'].isin(tickers_to_keep)].copy()
    
    # 6. Final validation
    final_nan_count = cleaned_data['Adj_Close'].isna().sum()
    if final_nan_count > 0:
        if verbose:
            print(f"   ⚠️  Warning: {final_nan_count} NaN prices still remain!")
        cleaned_data = cleaned_data.dropna(subset=['Adj_Close'])
    
    # 7. Sort by Ticker and Date for consistency
    cleaned_data = cleaned_data.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    if verbose:
        print(f"\n   📊 Final cleaned data:")
        print(f"      Tickers: {cleaned_data['Ticker'].nunique()}")
        print(f"      Date range: {cleaned_data['Date'].min()} to {cleaned_data['Date'].max()}")
        print(f"      Total rows: {len(cleaned_data):,}")
        print(f"      Avg observations per ticker: {len(cleaned_data) / cleaned_data['Ticker'].nunique():.1f}")
    
    # 8. Warning if too few tickers remain
    if len(tickers_to_keep) < 5:
        print(f"\n   ⚠️  WARNING: Only {len(tickers_to_keep)} tickers remain!")
        print(f"      This may be insufficient for robust portfolio analysis.")
        print(f"      Consider adjusting min_data_coverage (currently {min_data_coverage:.0%})")
    
    return cleaned_data, ticker_coverage

# Usage in your monthly loop:
def clean_data_pipeline_step(stage_1_portfolio_data, stage_1_benchmark_data, start_date, end_date):
    """
    Complete data cleaning step for the monthly loop
    """
    
    # Clean portfolio data
    stage_2_portfolio_data, portfolio_coverage = clean_portfolio_data_for_period(
        stage_1_portfolio_data=stage_1_portfolio_data,
        start_date=start_date,
        end_date=end_date,
        min_data_coverage=0.50,  # 50% coverage minimum for small universe
        verbose=True
    )
    
    # Clean benchmark data (simpler since it's usually just MTUM)
    stage_2_benchmark_data = stage_1_benchmark_data[
        (stage_1_benchmark_data['Date'] >= start_date) & 
        (stage_1_benchmark_data['Date'] <= end_date)
    ].copy()
    
    # Remove any NaN benchmark data
    initial_benchmark_nans = stage_2_benchmark_data['Adj_Close'].isna().sum()
    if initial_benchmark_nans > 0:
        print(f"   🔧 Removing {initial_benchmark_nans} NaN values from benchmark data")
        stage_2_benchmark_data = stage_2_benchmark_data.dropna(subset=['Adj_Close'])
    
    return stage_2_portfolio_data, stage_2_benchmark_data, portfolio_coverage

"""
# Replace your existing step 2 with:

###############################################################################################################################################################################
# 2.) Clean and Filter Data for 13 Month Analysis Period
###############################################################################################################################################################################

stage_2_portfolio_data, stage_2_benchmark_data, data_coverage_report = clean_data_pipeline_step(
    stage_1_portfolio_data=stage_1_portfolio_data,
    stage_1_benchmark_data=stage_1_benchmark_data, 
    start_date=start_date,
    end_date=end_date
)

# Check if we have enough data to proceed
if stage_2_portfolio_data['Ticker'].nunique() < 3:
    print(f"❌ Insufficient tickers ({stage_2_portfolio_data['Ticker'].nunique()}) for analysis in {d}")
    failed_optimizations.append({'date': d, 'reason': 'Insufficient tickers after data cleaning'})
    continue

print(f"✅ Data cleaning complete. Proceeding with {stage_2_portfolio_data['Ticker'].nunique()} tickers.")




"""
    