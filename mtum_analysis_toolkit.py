"""
MTUM Portfolio Analysis Toolkit - ENHANCED VERSION
==================================================

Professional analysis module for MTUM replica portfolio system.
NOW INCLUDES: Feature Importance Analysis and Visualization

Author: Portfolio Analysis System
Version: 1.1 - Enhanced with Feature Importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING AND FILE MANAGEMENT (Enhanced)
# =============================================================================

def find_latest_backtest_directory():
    """Find the most recent backtest results directory"""
    pattern = "backtest_results_*"
    directories = glob.glob(pattern)
    
    if not directories:
        # Also try monthly_backtest_results pattern
        pattern2 = "monthly_backtest_results_*"
        directories = glob.glob(pattern2)
        
        if not directories:
            print("⚠️ No backtest results directories found!")
            print("   Looking for patterns: backtest_results_* or monthly_backtest_results_*")
            return None
    
    latest_dir = sorted(directories)[-1]
    print(f"✅ Found latest backtest directory: {latest_dir}")
    return latest_dir

def find_csv_files_smart():
    """Smart CSV file finder that works with different naming patterns"""
    latest_dir = find_latest_backtest_directory()
    
    if not latest_dir:
        # Try finding files in current directory
        print("🔍 No directory found, searching current directory for CSV files...")
        csv_files = glob.glob("*.csv")
        
        if not csv_files:
            print("⚠️ No CSV files found in current directory")
            return {}
        
        # Parse CSV files in current directory
        file_paths = {}
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            
            # Map files based on content keywords
            if 'benchmark_comparison' in filename:
                file_paths['benchmark_comparison'] = csv_file
            elif 'daily_returns' in filename:
                file_paths['daily_returns'] = csv_file
            elif 'portfolio_composition' in filename:
                file_paths['portfolio_composition'] = csv_file
            elif 'portfolio_weights' in filename:
                file_paths['portfolio_weights'] = csv_file
            elif 'transaction_costs' in filename:
                file_paths['transaction_costs'] = csv_file
            elif 'prediction_accuracy' in filename:
                file_paths['prediction_accuracy'] = csv_file
        
        return file_paths
    
    # Search in directory structure
    file_paths = {}
    
    # Try organized subdirectory structure first
    csv_dir = os.path.join(latest_dir, "csv_files")
    if os.path.exists(csv_dir):
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    else:
        # Try flat directory structure
        csv_files = glob.glob(os.path.join(latest_dir, "*.csv"))
    
    # Parse found files
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        if 'benchmark_comparison' in filename:
            file_paths['benchmark_comparison'] = csv_file
        elif 'daily_returns' in filename:
            file_paths['daily_returns'] = csv_file
        elif 'portfolio_composition' in filename:
            file_paths['portfolio_composition'] = csv_file
        elif 'portfolio_weights' in filename:
            file_paths['portfolio_weights'] = csv_file
        elif 'transaction_costs' in filename:
            file_paths['transaction_costs'] = csv_file
        elif 'prediction_accuracy' in filename:
            file_paths['prediction_accuracy'] = csv_file
    
    return file_paths

def load_all_data():
    """Load all available data files"""
    print("🔍 Searching for data files...")
    file_paths = find_csv_files_smart()
    
    if not file_paths:
        print("⚠️ No data files found!")
        return {}
    
    print(f"📂 Found {len(file_paths)} data files:")
    for key, path in file_paths.items():
        print(f"   {key}: {os.path.basename(path)}")
    
    # Load data
    data = {}
    for key, file_path in file_paths.items():
        try:
            data[key] = pd.read_csv(file_path)
            print(f"✅ Loaded {key}: {len(data[key])} rows")
        except Exception as e:
            print(f"⚠️ Error loading {key}: {e}")
    
    return data

def reload_data():
    """Reload data if files have changed"""
    return load_all_data()

def quick_summary(data):
    """Quick summary of loaded data"""
    if data:
        print(f"📊 Loaded datasets: {list(data.keys())}")
        for key, df in data.items():
            print(f"   {key}: {df.shape}")
    else:
        print("⚠️ No data loaded")

# =============================================================================
# 2. FEATURE IMPORTANCE ANALYSIS FUNCTIONS (NEW!)
# =============================================================================

def parse_feature_importance_data(data):
    """Parse feature importance from prediction_accuracy CSV data"""
    
    if 'prediction_accuracy' not in data:
        print("⚠️ Prediction accuracy data not available for feature importance analysis")
        return None
    
    pa = data['prediction_accuracy'].copy()
    
    # Convert date column
    pa['date'] = pd.to_datetime(pa['date'])
    
    # Parse feature importance JSON strings
    feature_importance_list = []
    
    for idx, row in pa.iterrows():
        try:
            # Parse the JSON string in feature_importance column
            fi_dict = eval(row['feature_importance']) if isinstance(row['feature_importance'], str) else row['feature_importance']
            
            # Create record for this month
            fi_record = {'date': row['date']}
            fi_record.update(fi_dict)
            feature_importance_list.append(fi_record)
            
        except Exception as e:
            print(f"⚠️ Error parsing feature importance for {row['date']}: {e}")
            continue
    
    if not feature_importance_list:
        print("⚠️ No valid feature importance data found")
        return None
    
    # Convert to DataFrame
    fi_df = pd.DataFrame(feature_importance_list)
    fi_df.set_index('date', inplace=True)
    
    print(f"✅ Parsed feature importance data: {len(fi_df)} months, {len(fi_df.columns)} features")
    
    return fi_df

def get_top_features_over_time(feature_importance_df, top_n=10):
    """Get top N most important features over time"""
    
    if feature_importance_df is None:
        return None
    
    # Calculate average importance for each feature
    feature_means = feature_importance_df.mean().sort_values(ascending=False)
    top_features = feature_means.head(top_n)
    
    print(f"\n🏆 TOP {top_n} MOST IMPORTANT FEATURES (Average):")
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"   {i:2d}. {feature:<25} {importance:.4f}")
    
    return top_features

def analyze_feature_stability(feature_importance_df, top_n=10):
    """Analyze how stable feature importance rankings are over time"""
    
    if feature_importance_df is None:
        return None
    
    # Get overall top features
    feature_means = feature_importance_df.mean().sort_values(ascending=False)
    top_features = feature_means.head(top_n).index
    
    # Calculate statistics for top features
    stability_stats = {}
    
    for feature in top_features:
        values = feature_importance_df[feature]
        stability_stats[feature] = {
            'mean': values.mean(),
            'std': values.std(),
            'coefficient_of_variation': values.std() / values.mean() if values.mean() > 0 else np.inf,
            'min': values.min(),
            'max': values.max(),
            'trend': np.polyfit(range(len(values)), values, 1)[0]  # Linear trend slope
        }
    
    stability_df = pd.DataFrame(stability_stats).T
    stability_df = stability_df.sort_values('coefficient_of_variation')
    
    print(f"\n📈 FEATURE STABILITY ANALYSIS (Top {top_n}):")
    print("Features ranked by stability (lower CV = more stable):")
    print()
    for feature, stats in stability_df.head(10).iterrows():
        trend_direction = "📈" if stats['trend'] > 0 else "📉" if stats['trend'] < 0 else "➡️"
        print(f"   {feature:<25} CV: {stats['coefficient_of_variation']:.3f} {trend_direction}")
    
    return stability_df

# =============================================================================
# 3. FEATURE IMPORTANCE VISUALIZATION FUNCTIONS (NEW!)
# =============================================================================

def plot_feature_importance_evolution(data, top_n=8):
    """Plot how feature importance evolves over time"""
    
    # Parse feature importance data
    fi_df = parse_feature_importance_data(data)
    
    if fi_df is None:
        return
    
    # Get top features by average importance
    feature_means = fi_df.mean().sort_values(ascending=False)
    top_features = feature_means.head(top_n).index
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Evolution over time
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_features)))
    
    for i, feature in enumerate(top_features):
        ax1.plot(fi_df.index, fi_df[feature], 
                label=feature, linewidth=2, color=colors[i], marker='o', markersize=4)
    
    ax1.set_title(f'🚀 Feature Importance Evolution Over Time (Top {top_n})', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Feature Importance Score', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add trend lines for top 3 features
    for i, feature in enumerate(top_features[:3]):
        z = np.polyfit(range(len(fi_df)), fi_df[feature], 1)
        p = np.poly1d(z)
        ax1.plot(fi_df.index, p(range(len(fi_df))), 
                linestyle='--', color=colors[i], alpha=0.7, linewidth=1)
    
    # Plot 2: Current vs Historical Average
    current_importance = fi_df.iloc[-1][top_features]
    historical_avg = fi_df[top_features].mean()
    
    x_pos = np.arange(len(top_features))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, historical_avg, width, 
                    label='Historical Average', color='lightblue', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, current_importance, width, 
                    label='Latest Month', color='darkblue', alpha=0.8)
    
    ax2.set_title('📊 Current vs Historical Feature Importance', 
                  fontsize=16, fontweight='bold')
    ax2.set_ylabel('Feature Importance Score', fontsize=12)
    ax2.set_xlabel('Features', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(top_features, rotation=45, ha='right')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fi_df

def plot_feature_importance_heatmap(data, top_n=15):
    """Create a heatmap showing feature importance over time"""
    
    # Parse feature importance data
    fi_df = parse_feature_importance_data(data)
    
    if fi_df is None:
        return
    
    # Get top features and prepare data for heatmap
    feature_means = fi_df.mean().sort_values(ascending=False)
    top_features = feature_means.head(top_n).index
    
    heatmap_data = fi_df[top_features].T
    
    # Create the heatmap
    plt.figure(figsize=(16, 10))
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Feature Importance Score'},
                xticklabels=[d.strftime('%Y-%m') for d in fi_df.index[::6]], # Show every 6th month
                yticklabels=top_features,
                linewidths=0.5)
    
    plt.title(f'🔥 Feature Importance Heatmap Over Time (Top {top_n} Features)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Time Period', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance_rankings(data, top_n=10):
    """Plot how feature rankings change over time"""
    
    # Parse feature importance data
    fi_df = parse_feature_importance_data(data)
    
    if fi_df is None:
        return
    
    # Calculate monthly rankings
    monthly_rankings = fi_df.rank(axis=1, ascending=False)
    
    # Get features that appear in top N most frequently
    feature_means = fi_df.mean().sort_values(ascending=False)
    top_features = feature_means.head(top_n).index
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Ranking evolution (lower is better)
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_features)))
    
    for i, feature in enumerate(top_features):
        rankings = monthly_rankings[feature]
        ax1.plot(fi_df.index, rankings, 
                label=feature, linewidth=2, color=colors[i], 
                marker='o', markersize=3, alpha=0.8)
    
    ax1.set_title(f'📈 Feature Ranking Evolution (Top {top_n})', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Ranking (1 = Most Important)', fontsize=12)
    ax1.set_xlabel('Time Period', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Rank 1 at top
    ax1.set_ylim(top_n + 1, 0.5)
    
    # Plot 2: Average ranking distribution
    avg_rankings = monthly_rankings[top_features].mean().sort_values()
    
    bars = ax2.barh(range(len(avg_rankings)), avg_rankings.values, 
                    color='skyblue', edgecolor='navy', alpha=0.8)
    ax2.set_yticks(range(len(avg_rankings)))
    ax2.set_yticklabels(avg_rankings.index)
    ax2.set_xlabel('Average Ranking', fontsize=12)
    ax2.set_title('🏆 Average Feature Rankings', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add ranking values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance_current_analysis(data):
    """Create comprehensive analysis of current month's feature importance"""
    
    # Parse feature importance data
    fi_df = parse_feature_importance_data(data)
    
    if fi_df is None:
        return
    
    # Get current month data
    current_fi = fi_df.iloc[-1].sort_values(ascending=False)
    historical_avg = fi_df.mean().sort_values(ascending=False)
    
    # Calculate changes from historical average
    current_vs_avg = ((current_fi - historical_avg) / historical_avg * 100).fillna(0)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Current top features
    top_current = current_fi.head(12)
    bars1 = ax1.barh(range(len(top_current)), top_current.values, 
                     color='darkgreen', alpha=0.8)
    ax1.set_yticks(range(len(top_current)))
    ax1.set_yticklabels(top_current.index)
    ax1.set_xlabel('Feature Importance', fontsize=12)
    ax1.set_title('🎯 Current Month - Top Features', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # Plot 2: Historical average top features
    top_historical = historical_avg.head(12)
    bars2 = ax2.barh(range(len(top_historical)), top_historical.values, 
                     color='steelblue', alpha=0.8)
    ax2.set_yticks(range(len(top_historical)))
    ax2.set_yticklabels(top_historical.index)
    ax2.set_xlabel('Feature Importance', fontsize=12)
    ax2.set_title('📊 Historical Average - Top Features', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # Plot 3: Biggest increases
    biggest_increases = current_vs_avg.nlargest(10)
    colors_inc = ['darkgreen' if x > 0 else 'darkred' for x in biggest_increases.values]
    bars3 = ax3.barh(range(len(biggest_increases)), biggest_increases.values, 
                     color=colors_inc, alpha=0.8)
    ax3.set_yticks(range(len(biggest_increases)))
    ax3.set_yticklabels(biggest_increases.index)
    ax3.set_xlabel('% Change from Historical Average', fontsize=12)
    ax3.set_title('🚀 Biggest Increases This Month', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 4: Feature importance distribution
    ax4.hist(current_fi.values, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(current_fi.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {current_fi.mean():.3f}')
    ax4.axvline(current_fi.median(), color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {current_fi.median():.3f}')
    ax4.set_xlabel('Feature Importance Score', fontsize=12)
    ax4.set_ylabel('Number of Features', fontsize=12)
    ax4.set_title('📈 Current Month - Feature Importance Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary insights
    print("\n" + "="*60)
    print("🔍 FEATURE IMPORTANCE INSIGHTS")
    print("="*60)
    
    print(f"\n📅 Latest Analysis Date: {fi_df.index[-1].strftime('%B %Y')}")
    print(f"📊 Total Features Tracked: {len(current_fi)}")
    print(f"🏆 Most Important Feature: {current_fi.index[0]} ({current_fi.iloc[0]:.4f})")
    print(f"📈 Biggest Increase: {biggest_increases.index[0]} (+{biggest_increases.iloc[0]:.1f}%)")
    
    # Feature stability
    cv_scores = (fi_df.std() / fi_df.mean()).sort_values()
    print(f"🎯 Most Stable Feature: {cv_scores.index[0]} (CV: {cv_scores.iloc[0]:.3f})")
    print(f"🌪️  Most Volatile Feature: {cv_scores.index[-1]} (CV: {cv_scores.iloc[-1]:.3f})")

# =============================================================================
# 4. ENHANCED MODEL PERFORMANCE ANALYSIS (Updated)
# =============================================================================

def plot_model_performance(data):
    """Enhanced ML model performance visualization with feature importance"""
    
    if 'prediction_accuracy' not in data:
        print("⚠️ Prediction accuracy data not available")
        return
    
    pa = data['prediction_accuracy'].copy()
    pa['date'] = pd.to_datetime(pa['date'])
    
    # Parse feature importance for latest month
    fi_df = parse_feature_importance_data(data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot 1: Model accuracy metrics over time
    metrics_to_plot = ['accuracy_rate', 'precision_include', 'recall_include']
    colors = ['blue', 'green', 'orange']
    
    for metric, color in zip(metrics_to_plot, colors):
        if metric in pa.columns:
            axes[0].plot(pa['date'], pa[metric], 
                        marker='o', label=metric.replace('_', ' ').title(), 
                        linewidth=2, color=color, markersize=4)
    
    axes[0].set_title('🤖 Model Performance Over Time', fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: F1 Score trend
    if 'f1_score' in pa.columns:
        avg_f1 = pa['f1_score'].mean()
        axes[1].plot(pa['date'], pa['f1_score'], 
                    marker='o', color='purple', linewidth=2, markersize=4)
        axes[1].axhline(y=avg_f1, color='black', linestyle='--', alpha=0.7, 
                       label=f'Average: {avg_f1:.3f}')
        axes[1].set_title('🎯 F1 Score Evolution', fontweight='bold', fontsize=14)
        axes[1].set_ylabel('F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Current month feature importance (if available)
    if fi_df is not None:
        current_fi = fi_df.iloc[-1].sort_values(ascending=False).head(10)
        bars = axes[2].barh(range(len(current_fi)), current_fi.values, 
                           color='skyblue', edgecolor='navy')
        axes[2].set_yticks(range(len(current_fi)))
        axes[2].set_yticklabels(current_fi.index)
        axes[2].set_xlabel('Feature Importance')
        axes[2].set_title('🔥 Current Feature Importance (Top 10)', fontweight='bold', fontsize=14)
        axes[2].invert_yaxis()
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[2].text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
    else:
        axes[2].text(0.5, 0.5, 'Feature Importance\nData Not Available', 
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
    
    # Plot 4: Model performance distribution
    if 'accuracy_rate' in pa.columns:
        axes[3].hist(pa['accuracy_rate'], bins=15, alpha=0.7, color='lightgreen', 
                    edgecolor='darkgreen')
        axes[3].axvline(pa['accuracy_rate'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {pa["accuracy_rate"].mean():.3f}')
        axes[3].set_xlabel('Accuracy Rate')
        axes[3].set_ylabel('Frequency (Months)')
        axes[3].set_title('📊 Model Accuracy Distribution', fontweight='bold', fontsize=14)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 5. COMPREHENSIVE ANALYSIS FUNCTIONS (Updated)
# =============================================================================

def run_complete_analysis(data):
    """Run all analysis sections including feature importance"""
    
    if not data:
        print("⚠️ No data available for analysis")
        return
    
    print("\n" + "=" * 60)
    print("📊 RUNNING COMPLETE ANALYSIS WITH FEATURE IMPORTANCE")
    print("=" * 60)
    
    # 1. Performance charts
    print("\n📈 Generating performance charts...")
    plot_cumulative_performance(data)
    
    # 2. Portfolio characteristics
    print("\n🗂️ Analyzing portfolio characteristics...")
    plot_portfolio_characteristics(data)
    
    # 3. Portfolio weights
    print("\n⚖️ Examining portfolio weights...")
    plot_portfolio_weights(data)
    
    # 4. Enhanced model performance with feature importance
    print("\n🤖 Evaluating model performance...")
    plot_model_performance(data)
    
    # 5. NEW: Feature importance evolution
    print("\n🚀 Analyzing feature importance evolution...")
    plot_feature_importance_evolution(data)
    
    # 6. NEW: Feature importance heatmap
    print("\n🔥 Creating feature importance heatmap...")
    plot_feature_importance_heatmap(data)
    
    # 7. NEW: Feature rankings analysis
    print("\n🏆 Analyzing feature rankings...")
    plot_feature_importance_rankings(data)
    
    # 8. NEW: Current month feature analysis
    print("\n🎯 Current month feature importance analysis...")
    plot_feature_importance_current_analysis(data)
    
    # 9. Feature stability analysis
    print("\n📈 Feature stability analysis...")
    fi_df = parse_feature_importance_data(data)
    if fi_df is not None:
        analyze_feature_stability(fi_df)
        get_top_features_over_time(fi_df)
    
    # 10. Final summary table
    print("\n📋 Generating summary statistics...")
    create_summary_table(data)

# =============================================================================
# 6. ENHANCED REPORTING FUNCTIONS (Updated)
# =============================================================================

def generate_enhanced_executive_summary(data):
    """Enhanced executive summary including feature importance insights"""
    
    print("\n" + "=" * 60)
    print("📋 ENHANCED EXECUTIVE SUMMARY")
    print("=" * 60)
    
    summary = {}
    
    # Original summary logic (unchanged)
    analysis_start_date = None
    analysis_end_date = None
    analysis_period_days = None
    
    # Try to get date range from daily returns (most granular)
    if 'daily_returns' in data:
        dr = data['daily_returns']
        date_col = None
        for col in ['Date', 'date', 'dates']:
            if col in dr.columns:
                date_col = col
                break
        
        if date_col is not None:
            try:
                dates = pd.to_datetime(dr[date_col])
                analysis_start_date = dates.min()
                analysis_end_date = dates.max()
                analysis_period_days = (analysis_end_date - analysis_start_date).days
            except:
                pass
    
    # Display analysis period
    if analysis_start_date is not None and analysis_end_date is not None:
        print(f"🗓️ ANALYSIS PERIOD")
        print(f"   Start Date:                   {analysis_start_date.strftime('%B %d, %Y')}")
        print(f"   End Date:                     {analysis_end_date.strftime('%B %d, %Y')}")
        print(f"   Total Period:                 {analysis_period_days} days ({analysis_period_days/365.25:.1f} years)")
        
        # Calculate number of months for context
        months = len(data['benchmark_comparison']) if 'benchmark_comparison' in data else None
        if months:
            print(f"   Monthly Observations:         {months} months")
        
        summary['start_date'] = analysis_start_date
        summary['end_date'] = analysis_end_date
        summary['period_days'] = analysis_period_days
        summary['period_years'] = analysis_period_days/365.25
        
        print()  # Add spacing before performance metrics
    else:
        print("⚠️ Could not determine analysis period from data")
        print()
    
    # Benchmark comparison metrics
    if 'benchmark_comparison' in data:
        bc = data['benchmark_comparison']
        
        # Get latest metrics
        if len(bc) > 0:
            latest = bc.iloc[-1]
            
            summary['portfolio_return'] = latest.get('Portfolio_Annual_Return', 0)
            summary['benchmark_return'] = latest.get('Benchmark_Annual_Return', 0)
            summary['excess_return'] = latest.get('Excess_Annual_Return', 0)
            summary['tracking_error'] = bc['Tracking_Error'].mean() if 'Tracking_Error' in bc.columns else 0
            summary['info_ratio'] = bc['Information_Ratio'].mean() if 'Information_Ratio' in bc.columns else 0
            summary['sharpe_ratio'] = latest.get('Portfolio_Sharpe_Ratio', 0)
            summary['beta'] = latest.get('Beta', 0)
            summary['alpha'] = latest.get('Alpha_Annual', 0)
            summary['max_drawdown'] = latest.get('Portfolio_Max_Drawdown', 0)
            
            print(f"🎯 PERFORMANCE HIGHLIGHTS")
            print(f"   Portfolio Return (Latest):    {summary['portfolio_return']:.1%}")
            print(f"   Benchmark Return (Latest):    {summary['benchmark_return']:.1%}")
            print(f"   Excess Return:                {summary['excess_return']:.1%}")
            print(f"   Information Ratio:            {summary['info_ratio']:.3f}")
            print(f"   Sharpe Ratio:                 {summary['sharpe_ratio']:.3f}")
            print(f"   Beta:                         {summary['beta']:.3f}")
    
    # Portfolio characteristics
    if 'portfolio_composition' in data:
        pc = data['portfolio_composition']
        
        if len(pc) > 0:
            avg_holdings = pc['num_positions'].mean() if 'num_positions' in pc.columns else 0
            avg_concentration = pc['top5_concentration'].mean() if 'top5_concentration' in pc.columns else 0
            
            print(f"\n🗂️ PORTFOLIO STRUCTURE")
            print(f"   Average Holdings:             {avg_holdings:.1f} securities")
            print(f"   Top 5 Concentration:          {avg_concentration:.1%}")
            print(f"   vs MTUM Holdings:             200+ securities")
            
            summary['avg_holdings'] = avg_holdings
            summary['concentration'] = avg_concentration
    
    # Transaction costs
    if 'transaction_costs' in data:
        tc = data['transaction_costs']
        
        if len(tc) > 0:
            avg_turnover = tc['total_turnover'].mean() if 'total_turnover' in tc.columns else 0
            avg_cost_bps = tc['cost_bps'].mean() if 'cost_bps' in tc.columns else 0
            
            print(f"\n💰 TRANSACTION ANALYSIS")
            print(f"   Average Monthly Turnover:     {avg_turnover:.1%}")
            print(f"   Average Cost (bps):           {avg_cost_bps:.1f}")
            print(f"   Estimated Annual Cost:        {avg_cost_bps * 12:.0f} bps")
            
            summary['avg_turnover'] = avg_turnover
            summary['avg_cost_bps'] = avg_cost_bps
    
    # Model performance
    if 'prediction_accuracy' in data:
        pa = data['prediction_accuracy']
        
        if len(pa) > 0:
            avg_accuracy = pa['accuracy_rate'].mean() if 'accuracy_rate' in pa.columns else 0
            avg_f1 = pa['f1_score'].mean() if 'f1_score' in pa.columns else 0
            
            print(f"\n🤖 ML MODEL PERFORMANCE")
            print(f"   Average Accuracy:             {avg_accuracy:.1%}")
            print(f"   Average F1 Score:             {avg_f1:.3f}")
            
            summary['model_accuracy'] = avg_accuracy
            summary['model_f1'] = avg_f1
    
    # NEW: Feature Importance Analysis
    print(f"\n🚀 FEATURE IMPORTANCE INSIGHTS")
    fi_df = parse_feature_importance_data(data)
    
    if fi_df is not None:
        # Get top features
        top_features = get_top_features_over_time(fi_df, top_n=5)
        
        if top_features is not None:
            print(f"   Top Feature:                  {top_features.index[0]}")
            print(f"   Feature Importance:           {top_features.iloc[0]:.4f}")
            
            # Feature stability analysis
            stability_stats = analyze_feature_stability(fi_df, top_n=5)
            if stability_stats is not None:
                most_stable = stability_stats.index[0]
                cv_score = stability_stats.loc[most_stable, 'coefficient_of_variation']
                print(f"   Most Stable Feature:          {most_stable}")
                print(f"   Stability Score (CV):         {cv_score:.3f}")
                
                # Feature evolution
                current_fi = fi_df.iloc[-1].sort_values(ascending=False)
                historical_avg = fi_df.mean().sort_values(ascending=False)
                
                # Find features gaining importance
                current_vs_avg = ((current_fi - historical_avg) / historical_avg * 100).fillna(0)
                gaining_feature = current_vs_avg.nlargest(1)
                
                if len(gaining_feature) > 0:
                    print(f"   Rising Feature:               {gaining_feature.index[0]}")
                    print(f"   Importance Gain:              +{gaining_feature.iloc[0]:.1f}%")
                
                summary['top_feature'] = top_features.index[0]
                summary['top_feature_importance'] = top_features.iloc[0]
                summary['most_stable_feature'] = most_stable
                summary['feature_stability'] = cv_score
    else:
        print(f"   Feature importance data not available")
    
    return summary

def generate_executive_summary(data):
    """Generate executive summary from loaded data (ORIGINAL FUNCTION)"""
    
    print("\n" + "=" * 60)
    print("📋 EXECUTIVE SUMMARY")
    print("=" * 60)
    
    summary = {}
    
    # =============================================================================
    # ANALYSIS PERIOD DETECTION
    # =============================================================================
    
    analysis_start_date = None
    analysis_end_date = None
    analysis_period_days = None
    
    # Try to get date range from daily returns (most granular)
    if 'daily_returns' in data:
        dr = data['daily_returns']
        date_col = None
        for col in ['Date', 'date', 'dates']:
            if col in dr.columns:
                date_col = col
                break
        
        if date_col is not None:
            try:
                dates = pd.to_datetime(dr[date_col])
                analysis_start_date = dates.min()
                analysis_end_date = dates.max()
                analysis_period_days = (analysis_end_date - analysis_start_date).days
            except:
                pass
    
    # Fallback to benchmark comparison data
    if analysis_start_date is None and 'benchmark_comparison' in data:
        bc = data['benchmark_comparison']
        date_col = None
        for col in ['Date', 'date', 'dates']:
            if col in bc.columns:
                date_col = col
                break
        
        if date_col is not None:
            try:
                dates = pd.to_datetime(bc[date_col])
                analysis_start_date = dates.min()
                analysis_end_date = dates.max()
                analysis_period_days = (analysis_end_date - analysis_start_date).days
            except:
                pass
    
    # Display analysis period
    if analysis_start_date is not None and analysis_end_date is not None:
        print(f"🗓️ ANALYSIS PERIOD")
        print(f"   Start Date:                   {analysis_start_date.strftime('%B %d, %Y')}")
        print(f"   End Date:                     {analysis_end_date.strftime('%B %d, %Y')}")
        print(f"   Total Period:                 {analysis_period_days} days ({analysis_period_days/365.25:.1f} years)")
        
        # Calculate number of months for context
        months = len(data['benchmark_comparison']) if 'benchmark_comparison' in data else None
        if months:
            print(f"   Monthly Observations:         {months} months")
        
        summary['start_date'] = analysis_start_date
        summary['end_date'] = analysis_end_date
        summary['period_days'] = analysis_period_days
        summary['period_years'] = analysis_period_days/365.25
        
        print()  # Add spacing before performance metrics
    else:
        print("⚠️ Could not determine analysis period from data")
        print()
    
    # Benchmark comparison metrics
    if 'benchmark_comparison' in data:
        bc = data['benchmark_comparison']
        
        # Get latest metrics
        if len(bc) > 0:
            latest = bc.iloc[-1]
            
            summary['portfolio_return'] = latest.get('Portfolio_Annual_Return', 0)
            summary['benchmark_return'] = latest.get('Benchmark_Annual_Return', 0)
            summary['excess_return'] = latest.get('Excess_Annual_Return', 0)
            summary['tracking_error'] = bc['Tracking_Error'].mean() if 'Tracking_Error' in bc.columns else 0
            summary['info_ratio'] = bc['Information_Ratio'].mean() if 'Information_Ratio' in bc.columns else 0
            summary['sharpe_ratio'] = latest.get('Portfolio_Sharpe_Ratio', 0)
            summary['beta'] = latest.get('Beta', 0)
            summary['alpha'] = latest.get('Alpha_Annual', 0)
            summary['max_drawdown'] = latest.get('Portfolio_Max_Drawdown', 0)
            
            print(f"🎯 PERFORMANCE HIGHLIGHTS")
            print(f"   Portfolio Return (Latest):    {summary['portfolio_return']:.1%}")
            print(f"   Benchmark Return (Latest):    {summary['benchmark_return']:.1%}")
            print(f"   Excess Return:                {summary['excess_return']:.1%}")
            print(f"   Information Ratio:            {summary['info_ratio']:.3f}")
            print(f"   Sharpe Ratio:                 {summary['sharpe_ratio']:.3f}")
            print(f"   Beta:                         {summary['beta']:.3f}")
    
    # Portfolio characteristics
    if 'portfolio_composition' in data:
        pc = data['portfolio_composition']
        
        if len(pc) > 0:
            avg_holdings = pc['num_positions'].mean() if 'num_positions' in pc.columns else 0
            avg_concentration = pc['top5_concentration'].mean() if 'top5_concentration' in pc.columns else 0
            
            print(f"\n🗂️ PORTFOLIO STRUCTURE")
            print(f"   Average Holdings:             {avg_holdings:.1f} securities")
            print(f"   Top 5 Concentration:          {avg_concentration:.1%}")
            print(f"   vs MTUM Holdings:             200+ securities")
            
            summary['avg_holdings'] = avg_holdings
            summary['concentration'] = avg_concentration
    
    # Transaction costs
    if 'transaction_costs' in data:
        tc = data['transaction_costs']
        
        if len(tc) > 0:
            avg_turnover = tc['total_turnover'].mean() if 'total_turnover' in tc.columns else 0
            avg_cost_bps = tc['cost_bps'].mean() if 'cost_bps' in tc.columns else 0
            
            print(f"\n💰 TRANSACTION ANALYSIS")
            print(f"   Average Monthly Turnover:     {avg_turnover:.1%}")
            print(f"   Average Cost (bps):           {avg_cost_bps:.1f}")
            print(f"   Estimated Annual Cost:        {avg_cost_bps * 12:.0f} bps")
            
            summary['avg_turnover'] = avg_turnover
            summary['avg_cost_bps'] = avg_cost_bps
    
    # Model performance
    if 'prediction_accuracy' in data:
        pa = data['prediction_accuracy']
        
        if len(pa) > 0:
            avg_accuracy = pa['accuracy_rate'].mean() if 'accuracy_rate' in pa.columns else 0
            avg_f1 = pa['f1_score'].mean() if 'f1_score' in pa.columns else 0
            
            print(f"\n🤖 ML MODEL PERFORMANCE")
            print(f"   Average Accuracy:             {avg_accuracy:.1%}")
            print(f"   Average F1 Score:             {avg_f1:.3f}")
            
            summary['model_accuracy'] = avg_accuracy
            summary['model_f1'] = avg_f1
    
    return summary

def get_portfolio_stats_table(data):
    """Get a formatted table of portfolio statistics"""
    if not data:
        return pd.DataFrame()
    
    stats = {}
    
    # From benchmark comparison
    if 'benchmark_comparison' in data:
        bc = data['benchmark_comparison']
        if len(bc) > 0:
            latest = bc.iloc[-1]
            stats.update({
                'Portfolio Return': f"{latest.get('Portfolio_Annual_Return', 0):.2%}",
                'Benchmark Return': f"{latest.get('Benchmark_Annual_Return', 0):.2%}",
                'Tracking Error': f"{bc['Tracking_Error'].mean():.2%}",
                'Information Ratio': f"{bc['Information_Ratio'].mean():.3f}",
                'Sharpe Ratio': f"{latest.get('Portfolio_Sharpe_Ratio', 0):.3f}",
                'Beta': f"{latest.get('Beta', 0):.3f}",
                'Alpha': f"{latest.get('Alpha_Annual', 0):.2%}"
            })
    
    # From portfolio composition
    if 'portfolio_composition' in data:
        pc = data['portfolio_composition']
        if len(pc) > 0:
            stats.update({
                'Avg Holdings': f"{pc['num_positions'].mean():.1f}",
                'Concentration': f"{pc['top5_concentration'].mean():.1%}"
            })
    
    # From transaction costs
    if 'transaction_costs' in data:
        tc = data['transaction_costs']
        if len(tc) > 0:
            stats.update({
                'Monthly Turnover': f"{tc['total_turnover'].mean():.1%}",
                'Cost per Month': f"{tc['cost_bps'].mean():.1f} bps"
            })
    
    # Convert to DataFrame for nice display
    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    return stats_df

def analyze_specific_metric(data, metric_name):
    """Analyze a specific metric across time"""
    for key, df in data.items():
        if metric_name in df.columns:
            print(f"\n📊 {metric_name} in {key}:")
            print(f"   Mean: {df[metric_name].mean():.4f}")
            print(f"   Std:  {df[metric_name].std():.4f}")
            print(f"   Min:  {df[metric_name].min():.4f}")
            print(f"   Max:  {df[metric_name].max():.4f}")
            
            # Plot if numerical
            if df[metric_name].dtype in ['float64', 'int64']:
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df[metric_name], marker='o', linewidth=2)
                plt.title(f'{metric_name} Over Time')
                plt.ylabel(metric_name)
                plt.grid(True, alpha=0.3)
                plt.show()

def compare_returns_detailed(data):
    """Detailed return comparison analysis"""
    if 'daily_returns' not in data:
        print("⚠️ Daily returns data not available")
        return
    
    df = data['daily_returns'].copy()
    
    # Handle date column
    date_col = None
    for col in ['Date', 'date', 'dates']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    
    # Get return columns
    port_col = 'portfolio_return' if 'portfolio_return' in df.columns else df.columns[0]
    bench_col = 'benchmark_return' if 'benchmark_return' in df.columns else df.columns[1]
    
    # Calculate statistics
    port_returns = df[port_col]
    bench_returns = df[bench_col]
    excess_returns = port_returns - bench_returns
    
    print("📊 DETAILED RETURN ANALYSIS")
    print("=" * 40)
    print(f"Portfolio Returns:")
    print(f"   Annualized Return: {port_returns.mean() * 252:.2%}")
    print(f"   Volatility:        {port_returns.std() * np.sqrt(252):.2%}")
    print(f"   Sharpe Ratio:      {(port_returns.mean() / port_returns.std()) * np.sqrt(252):.3f}")
    
    print(f"\nBenchmark Returns:")
    print(f"   Annualized Return: {bench_returns.mean() * 252:.2%}")
    print(f"   Volatility:        {bench_returns.std() * np.sqrt(252):.2%}")
    print(f"   Sharpe Ratio:      {(bench_returns.mean() / bench_returns.std()) * np.sqrt(252):.3f}")
    
    print(f"\nExcess Returns:")
    print(f"   Annualized:        {excess_returns.mean() * 252:.2%}")
    print(f"   Tracking Error:    {excess_returns.std() * np.sqrt(252):.2%}")
    print(f"   Information Ratio: {(excess_returns.mean() / excess_returns.std()) * np.sqrt(252):.3f}")
    
    # Distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Return distributions
    ax1.hist(port_returns * 100, bins=30, alpha=0.7, label='Portfolio', color='blue')
    ax1.hist(bench_returns * 100, bins=30, alpha=0.7, label='Benchmark', color='orange')
    ax1.set_title('Daily Return Distributions')
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Excess return distribution
    ax2.hist(excess_returns * 100, bins=30, alpha=0.7, color='green')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Excess Return Distribution')
    ax2.set_xlabel('Excess Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def export_summary_to_excel(data, filename='mtum_analysis_summary.xlsx'):
    """Export all analysis to Excel file"""
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for key, df in data.items():
                # Clean sheet name (Excel has character limits)
                sheet_name = key[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"✅ Data exported to {filename}")
            print(f"   Sheets created: {list(data.keys())}")
    except Exception as e:
        print(f"⚠️ Error exporting to Excel: {e}")

def generate_markdown_executive_report(data, summary_metrics, filename='executive_analysis_report.md'):
    """Generate a comprehensive markdown executive report with dates"""
    
    if not data or not summary_metrics:
        print("⚠️ Cannot generate report without data and summary metrics")
        return
    
    # Get analysis period
    start_date = summary_metrics.get('start_date')
    end_date = summary_metrics.get('end_date')
    period_years = summary_metrics.get('period_years', 0)
    
    # Generate timestamp
    report_generated = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    
    # Create markdown content
    markdown_content = f"""# MTUM Replica Portfolio System
## Executive Analysis Report

**Report Generated:** {report_generated}

---

## 📊 Analysis Period

**Backtest Timeline:** {start_date.strftime('%B %d, %Y') if start_date else 'N/A'} to {end_date.strftime('%B %d, %Y') if end_date else 'N/A'}

**Total Duration:** {period_years:.1f} years ({summary_metrics.get('period_days', 0)} days)

**Monthly Observations:** {len(data.get('benchmark_comparison', []))} months

---

## 🎯 Executive Summary

### Key Achievement
Successfully replicated MTUM (iShares MSCI USA Momentum Factor ETF) performance using a **sparse portfolio of {summary_metrics.get('avg_holdings', 'N/A'):.1f} securities** instead of MTUM's 200+ holdings, while maintaining similar risk-return characteristics.

### Performance Highlights

| Metric | Portfolio | Benchmark (MTUM) | 
|--------|-----------|------------------|
| **Annual Return** | {summary_metrics.get('portfolio_return', 0):.1%} | {summary_metrics.get('benchmark_return', 0):.1%} |
| **Excess Return** | {summary_metrics.get('excess_return', 0):.1%} | - |
| **Sharpe Ratio** | {summary_metrics.get('sharpe_ratio', 0):.3f} | - |
| **Information Ratio** | {summary_metrics.get('info_ratio', 0):.3f} | - |
| **Beta** | {summary_metrics.get('beta', 0):.3f} | 1.00 |
| **Tracking Error** | {summary_metrics.get('tracking_error', 0):.1%} | - |

### Portfolio Characteristics

| Characteristic | Value | Comparison |
|----------------|--------|------------|
| **Average Holdings** | {summary_metrics.get('avg_holdings', 0):.1f} securities | vs 200+ in MTUM |
| **Top 5 Concentration** | {summary_metrics.get('concentration', 0):.1%} | Focused exposure |
| **Monthly Turnover** | {summary_metrics.get('avg_turnover', 0):.1%} | Moderate trading |
| **Transaction Costs** | {summary_metrics.get('avg_cost_bps', 0) * 12:.0f} bps annually | Cost-efficient |

---

## 🏆 Key Achievements

✅ **Sparse Replication Success:** Achieved similar performance with {((200 - summary_metrics.get('avg_holdings', 0)) / 200 * 100):.0f}% fewer securities

✅ **Active Management Value:** Information Ratio of {summary_metrics.get('info_ratio', 0):.3f} {"(excellent)" if summary_metrics.get('info_ratio', 0) > 0.5 else "(solid)"}

✅ **Risk Management:** Controlled tracking error at {summary_metrics.get('tracking_error', 0):.1%}

✅ **Systematic Approach:** Reproducible ML-driven methodology

---

## 💡 Business Impact

### Operational Advantages
- **Reduced Complexity:** Easier portfolio monitoring and management
- **Lower Costs:** Fewer holdings reduce operational overhead
- **Scalability:** Systematic process enables easy replication
- **Risk Control:** Systematic tracking error management

### Cost Analysis
"""

    # Add cost analysis if available
    if summary_metrics.get('avg_cost_bps') and summary_metrics.get('alpha'):
        annual_cost = summary_metrics['avg_cost_bps'] * 12
        alpha = summary_metrics['alpha']
        if annual_cost > 0 and alpha > 0:
            cost_coverage = (alpha * 10000) / annual_cost
            markdown_content += f"""
- **Annual Transaction Costs:** ~{annual_cost:.0f} basis points
- **Alpha Generation:** {alpha:.1%} annually  
- **Cost Coverage:** Alpha covers costs by {cost_coverage:.1f}x
"""
        else:
            markdown_content += "\n- Cost analysis data not available\n"
    
    # Add methodology section
    markdown_content += f"""
---

## 🔬 Methodology

### 1. Machine Learning Security Selection
- **XGBoost Classification** for ETF inclusion decisions
- **Features:** Momentum (1M, 3M, 6M, 12M), volatility, correlation with MTUM
- **Validation:** Time-series cross-validation to prevent look-ahead bias

### 2. Portfolio Optimization  
- **CVXPY mean-variance optimization** with L1 regularization
- **Constraints:** Long-only, weight limits, sparsity enforcement
- **Objective:** Minimize tracking error while maintaining momentum exposure

### 3. Performance Evaluation
- **Walk-forward backtesting** with {summary_metrics.get('period_years', 0):.1f} years of data
- **Comprehensive metrics:** Information ratio, tracking error, transaction costs
- **Benchmark comparison** against MTUM performance

---

## 📈 Model Performance

"""

    # Add model performance if available
    if 'prediction_accuracy' in data:
        pa = data['prediction_accuracy']
        avg_accuracy = pa['accuracy_rate'].mean() if 'accuracy_rate' in pa.columns else 0
        avg_f1 = pa['f1_score'].mean() if 'f1_score' in pa.columns else 0
        
        markdown_content += f"""
| ML Metric | Value | Assessment |
|-----------|-------|------------|
| **Average Accuracy** | {avg_accuracy:.1%} | {"Good" if avg_accuracy > 0.6 else "Moderate"} |
| **F1 Score** | {avg_f1:.3f} | {"Strong" if avg_f1 > 0.6 else "Moderate"} |
| **Prediction Stability** | Consistent across {len(pa)} months | Reliable |
"""
    else:
        markdown_content += "Model performance data not available in current dataset.\n"

    # Add conclusions
    markdown_content += f"""
---

## 🎯 Conclusions & Next Steps

### Strategic Value
This system demonstrates how **advanced quantitative techniques** can create operational advantages while maintaining investment objectives. The {summary_metrics.get('avg_holdings', 0):.1f}-security portfolio provides similar momentum factor exposure to MTUM with significantly reduced complexity.

### Technical Excellence
- **End-to-end systematic process** from data to portfolio construction
- **Machine learning integration** with traditional optimization methods  
- **Production-ready implementation** with proper validation and risk controls
- **Institutional-quality performance analysis** and reporting

### Recommended Applications
1. **Factor ETF replication** for other style factors (value, quality, low volatility)
2. **Cost reduction initiatives** for large institutional portfolios
3. **Custom benchmark creation** for specific investment mandates
4. **Risk management enhancement** through systematic tracking error control

---

## 📋 Technical Specifications

**Analysis Period:** {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}

**Data Sources:** 11 sector ETFs, MTUM benchmark, daily price data

**Rebalancing Frequency:** Monthly

**Technology Stack:** Python, XGBoost, CVXPY, Pandas, NumPy

**Validation Method:** Walk-forward backtesting with time-series cross-validation

---

*Report auto-generated from backtest results on {report_generated}*
"""

    # Write to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"✅ Executive report generated: {filename}")
        print(f"📅 Analysis period: {start_date.strftime('%B %d, %Y') if start_date else 'N/A'} to {end_date.strftime('%B %d, %Y') if end_date else 'N/A'}")
        print(f"📊 Duration: {period_years:.1f} years")
        
    except Exception as e:
        print(f"⚠️ Error generating report: {e}")
        
    return markdown_content

def print_business_conclusions(data, summary_metrics):
    """Print business impact and conclusions"""
    
    print("\n" + "=" * 60)
    print("🎯 BUSINESS IMPACT & CONCLUSIONS")
    print("=" * 60)
    
    if not summary_metrics:
        print("⚠️ Limited conclusions available without complete data")
        return
    
    # Key achievements
    info_ratio = summary_metrics.get('info_ratio', 0)
    avg_holdings = summary_metrics.get('avg_holdings', 0)
    alpha = summary_metrics.get('alpha', 0)
    beta = summary_metrics.get('beta', 0)
    
    print("🏆 KEY ACHIEVEMENTS:")
    if avg_holdings > 0:
        print(f"   ✅ Successfully replicated MTUM with {avg_holdings:.0f} securities vs 200+")
    if info_ratio > 0:
        print(f"   ✅ Achieved {info_ratio:.3f} Information Ratio", end="")
        if info_ratio > 0.5:
            print(" (>0.5 is excellent active management)")
        else:
            print(" (solid active management)")
    if alpha != 0:
        print(f"   ✅ Generated {alpha:.1%} annual alpha")
    if beta > 0:
        print(f"   ✅ Maintained {beta:.2f} beta exposure to market")
    
    print("\n💡 OPERATIONAL ADVANTAGES:")
    print("   ✅ Reduced complexity: easier to monitor and manage")
    print("   ✅ Lower operational costs: fewer holdings = lower fees")
    print("   ✅ Systematic approach: reproducible and scalable")
    print("   ✅ Risk management: controlled tracking error")
    
    # Cost analysis
    if 'avg_cost_bps' in summary_metrics and alpha > 0:
        annual_cost = summary_metrics['avg_cost_bps'] * 12
        if annual_cost > 0:
            cost_coverage = (alpha * 10000) / annual_cost
            print(f"\n💰 COST ANALYSIS:")
            print(f"   📊 Annual transaction costs: ~{annual_cost:.0f} basis points")
            print(f"   📈 Alpha generation: {alpha:.1%}")
            print(f"   ⚖️ Alpha covers costs by: {cost_coverage:.1f}x")
    
    print("\n🚀 INTERVIEW TALKING POINTS:")
    print("   • 'Built end-to-end systematic portfolio management system'")
    print("   • 'Combined ML and optimization for sparse factor replication'")
    print("   • 'Achieved institutional-quality performance with operational advantages'")
    print("   • 'Demonstrated both technical depth and business understanding'")
    
    print(f"\n{'=' * 60}")
    print("✅ ANALYSIS COMPLETE - 🎯")
    print(f"{'=' * 60}")

# =============================================================================
# 7. ORIGINAL FUNCTIONS (Unchanged - keeping all existing functionality)
# =============================================================================

def create_summary_table(data):
    """Create comprehensive summary table"""
    
    if 'benchmark_comparison' not in data:
        print("⚠️ Benchmark comparison data not available for summary table")
        return
    
    bc = data['benchmark_comparison']
    latest = bc.iloc[-1] if len(bc) > 0 else {}
    
    # Portfolio characteristics
    avg_holdings = "N/A"
    concentration = "N/A"
    if 'portfolio_composition' in data:
        pc = data['portfolio_composition']
        if len(pc) > 0:
            avg_holdings = f"{pc['num_positions'].mean():.1f}"
            if 'top5_concentration' in pc.columns:
                concentration = f"{pc['top5_concentration'].mean():.1%}"
    
    # Transaction costs
    annual_turnover = "N/A"
    annual_cost = "N/A"
    if 'transaction_costs' in data:
        tc = data['transaction_costs']
        if len(tc) > 0:
            monthly_turnover = tc['total_turnover'].mean()
            annual_turnover = f"{monthly_turnover * 12:.0%}"
            if 'cost_bps' in tc.columns:
                annual_cost = f"{tc['cost_bps'].mean() * 12:.0f} bps"
    
    # Create summary
    summary_data = {
        'Metric': [
            'Portfolio Return (Annual)', 'Benchmark Return (Annual)', 'Excess Return',
            'Tracking Error', 'Information Ratio', 'Sharpe Ratio',
            'Beta', 'Alpha (Annual)', 'Max Drawdown',
            'Average Holdings', 'Top 5 Concentration', 'Annual Turnover', 'Est. Annual Costs'
        ],
        'Value': [
            f"{latest.get('Portfolio_Annual_Return', 0):.1%}",
            f"{latest.get('Benchmark_Annual_Return', 0):.1%}",
            f"{latest.get('Excess_Annual_Return', 0):.1%}",
            f"{bc['Tracking_Error'].mean() if 'Tracking_Error' in bc.columns else 0:.1%}",
            f"{bc['Information_Ratio'].mean() if 'Information_Ratio' in bc.columns else 0:.3f}",
            f"{latest.get('Portfolio_Sharpe_Ratio', 0):.3f}",
            f"{latest.get('Beta', 0):.3f}",
            f"{latest.get('Alpha_Annual', 0):.1%}",
            f"{latest.get('Portfolio_Max_Drawdown', 0):.1%}",
            avg_holdings, concentration, annual_turnover, annual_cost
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n📊 COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 50)
    print(summary_df.to_string(index=False))

def plot_cumulative_performance(data):
    """Plot cumulative returns comparison"""
    if 'daily_returns' not in data:
        print("⚠️ Daily returns data not available for performance chart")
        return
    
    df = data['daily_returns'].copy()
    
    # Handle different possible date column names
    date_col = None
    for col in ['Date', 'date', 'dates']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("⚠️ No date column found in daily returns data")
        return
    
    # Convert date and set index
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    # Calculate cumulative returns
    port_return_col = 'portfolio_return' if 'portfolio_return' in df.columns else df.columns[0]
    bench_return_col = 'benchmark_return' if 'benchmark_return' in df.columns else df.columns[1]
    
    df['Portfolio_Cumulative'] = (1 + df[port_return_col]).cumprod()
    df['Benchmark_Cumulative'] = (1 + df[bench_return_col]).cumprod()
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Cumulative returns
    ax1.plot(df.index, df['Portfolio_Cumulative'], 
             label='Portfolio', linewidth=3, color='#2563eb')
    ax1.plot(df.index, df['Benchmark_Cumulative'], 
             label='MTUM Benchmark', linewidth=3, color='#f59e0b')
    
    # Add performance stats
    total_port_return = df['Portfolio_Cumulative'].iloc[-1] - 1
    total_bench_return = df['Benchmark_Cumulative'].iloc[-1] - 1
    outperformance = total_port_return - total_bench_return
    
    ax1.text(0.02, 0.98, 
             f'Total Return:\nPortfolio: {total_port_return:.1%}\nMTUM: {total_bench_return:.1%}\nOutperformance: {outperformance:.1%}', 
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax1.set_title('📈 Cumulative Returns: Portfolio vs MTUM Benchmark', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Tracking error if available
    if 'benchmark_comparison' in data:
        bc = data['benchmark_comparison'].copy()
        bc['date'] = pd.to_datetime(bc['date'] if 'date' in bc.columns else bc.index)
        
        ax2.plot(bc['date'], bc['Tracking_Error'] * 100, 
                color='red', linewidth=2, marker='o', markersize=6)
        ax2.axhline(y=bc['Tracking_Error'].mean() * 100, color='black', linestyle='--', alpha=0.7, 
                   label=f'Average: {bc["Tracking_Error"].mean():.1%}')
        
        ax2.set_title('🎯 Monthly Tracking Error Evolution', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Tracking Error (%)')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_portfolio_characteristics(data):
    """Plot portfolio characteristics over time"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_count = 0
    
    # Portfolio composition
    if 'portfolio_composition' in data:
        pc = data['portfolio_composition'].copy()
        pc['date'] = pd.to_datetime(pc['date'] if 'date' in pc.columns else pc.index)
        
        # Number of holdings
        if 'num_positions' in pc.columns:
            axes[plot_count].plot(pc['date'], pc['num_positions'], 
                                 marker='o', linewidth=2, color='blue', markersize=8)
            axes[plot_count].set_title('🗂️ Number of Holdings Over Time', fontweight='bold')
            axes[plot_count].set_ylabel('Number of Securities')
            axes[plot_count].grid(True, alpha=0.3)
            plot_count += 1
        
        # Concentration
        if 'top5_concentration' in pc.columns:
            axes[plot_count].bar(range(len(pc)), pc['top5_concentration'] * 100, 
                                color='lightcoral', alpha=0.7)
            axes[plot_count].set_title('🎯 Top 5 Holdings Concentration', fontweight='bold')
            axes[plot_count].set_ylabel('Concentration (%)')
            axes[plot_count].set_xlabel('Month')
            axes[plot_count].grid(True, alpha=0.3)
            plot_count += 1
    
    # Transaction costs
    if 'transaction_costs' in data and plot_count < 4:
        tc = data['transaction_costs'].copy()
        
        # Turnover
        if 'total_turnover' in tc.columns:
            axes[plot_count].bar(range(len(tc)), tc['total_turnover'] * 100, 
                                color='lightgreen', alpha=0.7)
            axes[plot_count].set_title('🔄 Monthly Portfolio Turnover', fontweight='bold')
            axes[plot_count].set_ylabel('Turnover (%)')
            axes[plot_count].set_xlabel('Month')
            axes[plot_count].grid(True, alpha=0.3)
            plot_count += 1
        
        # Costs
        if 'cost_bps' in tc.columns and plot_count < 4:
            axes[plot_count].bar(range(len(tc)), tc['cost_bps'], 
                                color='gold', alpha=0.7)
            axes[plot_count].set_title('💰 Monthly Transaction Costs', fontweight='bold')
            axes[plot_count].set_ylabel('Cost (basis points)')
            axes[plot_count].set_xlabel('Month')
            axes[plot_count].grid(True, alpha=0.3)
            plot_count += 1
    
    # Hide unused subplots
    for i in range(plot_count, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_portfolio_weights(data):
    """Plot current portfolio weights and evolution"""
    if 'portfolio_weights' not in data:
        print("⚠️ Portfolio weights data not available")
        return
    
    pw = data['portfolio_weights'].copy()
    
    # Handle date column
    date_col = 'Date' if 'Date' in pw.columns else 'date'
    if date_col in pw.columns:
        pw[date_col] = pd.to_datetime(pw[date_col])
        pw.set_index(date_col, inplace=True)
    
    # Get latest weights
    latest_weights = pw.iloc[-1]
    weight_cols = [col for col in pw.columns if col not in ['Date', 'date']]
    
    # Find current positions (non-zero weights)
    current_positions = {}
    for col in weight_cols:
        if pd.notna(latest_weights[col]) and latest_weights[col] > 0:
            current_positions[col] = latest_weights[col]
    
    if not current_positions:
        print("⚠️ No current positions found")
        return
    
    print(f"\n📋 CURRENT PORTFOLIO COMPOSITION (Latest Month):")
    for etf, weight in sorted(current_positions.items(), key=lambda x: x[1], reverse=True):
        print(f"   {etf}: {weight:.1%}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of current allocation
    etfs = list(current_positions.keys())
    weights = list(current_positions.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(etfs)))
    ax1.pie(weights, labels=etfs, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('🥧 Current Portfolio Allocation', fontweight='bold', fontsize=14)
    
    # Weight evolution over time
    for etf in etfs:
        if etf in pw.columns:
            weights_series = pw[etf].fillna(0)
            ax2.plot(pw.index, weights_series * 100, 
                    marker='o', label=etf, linewidth=2, markersize=4)
    
    ax2.set_title('📈 Portfolio Weights Evolution', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Weight (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 8. ENHANCED MODULE INFORMATION (Updated)
# =============================================================================

__version__ = "1.1.0"
__author__ = "Portfolio Analysis System"
__description__ = "Enhanced toolkit for MTUM replica portfolio analysis with Feature Importance"

def print_module_info():
    """Print enhanced module information and available functions"""
    
    print("=" * 70)
    print("📦 ENHANCED MTUM ANALYSIS TOOLKIT")
    print("=" * 70)
    print(f"Version: {__version__}")
    print(f"Description: {__description__}")
    print()
    
    print("🔧 AVAILABLE FUNCTIONS:")
    print()
    print("📁 Data Loading:")
    print("   • load_all_data() - Automatically find and load backtest data")
    print("   • reload_data() - Refresh data from latest backtest")
    print("   • quick_summary(data) - Show loaded datasets overview")
    print()
    print("📊 Analysis & Summary:")
    print("   • generate_enhanced_executive_summary(data) - Complete summary with feature insights")
    print("   • create_summary_table(data) - Formatted performance table")
    print("   • get_portfolio_stats_table(data) - Statistics DataFrame")
    print()
    print("🚀 NEW: Feature Importance Analysis:")
    print("   • parse_feature_importance_data(data) - Parse feature importance from CSV")
    print("   • plot_feature_importance_evolution(data) - Feature importance over time")
    print("   • plot_feature_importance_heatmap(data) - Heatmap visualization")
    print("   • plot_feature_importance_rankings(data) - Feature ranking changes")
    print("   • plot_feature_importance_current_analysis(data) - Current month deep dive")
    print("   • analyze_feature_stability(fi_df) - Feature stability statistics")
    print("   • get_top_features_over_time(fi_df) - Top features analysis")
    print()
    print("📈 Visualization:")
    print("   • plot_cumulative_performance(data) - Returns comparison charts")
    print("   • plot_portfolio_characteristics(data) - Portfolio metrics over time")
    print("   • plot_portfolio_weights(data) - Current and historical weights")
    print("   • plot_model_performance(data) - Enhanced ML model metrics with feature importance")
    print()
    print("🔍 Advanced Analysis:")
    print("   • run_complete_analysis(data) - Full analysis suite with feature importance")
    print("   • analyze_specific_metric(data, metric) - Deep dive on one metric")
    print("   • compare_returns_detailed(data) - Statistical return analysis")
    print()
    print("📄 Reporting & Export:")
    print("   • generate_markdown_executive_report(data, summary, filename)")
    print("   • export_summary_to_excel(data, filename)")
    print("   • print_business_conclusions(data, summary)")
    print()
    print("💡 Enhanced Usage Example:")
    print("   import mtum_analysis_toolkit as mtum")
    print("   data = mtum.load_all_data()")
    print("   summary = mtum.generate_enhanced_executive_summary(data)")
    print("   mtum.run_complete_analysis(data)  # Now includes feature importance!")
    print("   mtum.plot_feature_importance_evolution(data)")
    print("=" * 70)

# =============================================================================
# 9. ENHANCED EXPORT SHORTCUTS (Updated)
# =============================================================================

# Enhanced workflow functions for easy access
__all__ = [
    # Core workflow (BOTH original and enhanced)
    'load_all_data', 'generate_executive_summary', 'generate_enhanced_executive_summary', 'run_complete_analysis',
    
    # NEW: Feature Importance Functions
    'parse_feature_importance_data', 'plot_feature_importance_evolution',
    'plot_feature_importance_heatmap', 'plot_feature_importance_rankings',
    'plot_feature_importance_current_analysis', 'analyze_feature_stability',
    'get_top_features_over_time',
    
    # Enhanced Visualization
    'plot_cumulative_performance', 'plot_portfolio_characteristics', 
    'plot_portfolio_weights', 'plot_model_performance',
    
    # Analysis (RESTORED originals)
    'create_summary_table', 'analyze_specific_metric', 'compare_returns_detailed',
    'get_portfolio_stats_table',
    
    # Export & Reporting
    'generate_markdown_executive_report', 'export_summary_to_excel', 
    'print_business_conclusions',
    
    # Utilities
    'quick_summary', 'reload_data', 'print_module_info'
]

# Initialize matplotlib style when module is imported
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("🚀 Enhanced MTUM Analysis Toolkit v1.1 loaded with Feature Importance capabilities!")
print("📊 Use run_complete_analysis(data) for comprehensive analysis including feature importance")
print("🔥 New functions: plot_feature_importance_evolution(), plot_feature_importance_heatmap(), and more!")