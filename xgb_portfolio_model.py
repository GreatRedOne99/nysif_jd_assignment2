import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
#from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve,  accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, roc_curve, 
    accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

"""
### Calculating Tracking Error (Label Generation):
      > For each security, calculate the historical difference between its return and the benchmark return.
      Based on these differences, define a threshold to classify securities as either
      "include" (low tracking error) or "exclude" (high tracking error).
      This threshold needs to be determined based on the specific investment strategy and risk tolerance.
      For example, you might include securities within a certain standard deviation of the benchmark return.
      # For Target Label Creation for XGBoost Classifier to include / exclude an ETF

      threshold = stage_7_portfolio_data['monthly_tracking_error'].mean() + stage_7_portfolio_data['monthly_tracking_error'].std()

      monthly_data_annualized_tracking_error

      monthly_stage_1_portfolio_data.groupby(['Ticker'])['ln_change_differences'].resample('ME', level='Date').apply(lambda s: s.std() * np.sqrt(trading_days_per_year)).rename('monthly_tracking_error').reset_index(drop=False)

      Threshold for the tracking error from the data is the mean of the MONTHLY Tracking Error + 1 the Standard Deviation of the Monthly Tracking Error



Turns out this function is the key to include or exclude securities in the beginning!!


"""
def binary_classifier(data, threshold=0.5):
  # Define target variable categories - Example: Binary classification based on a threshold
  # You might need to adjust the threshold based on your analysis of tracking error distribution
  #print(data)
  if abs(data) <= threshold:  # Example threshold: within 0.5% tracking error
      return 1 # Include
  else:
      return 0 # Exclude


def prepare_next_month_target(df, date_col='date', security_col='security', target_col='include_exclude'):
    """
    Create target for predicting NEXT month's portfolio decisions
    
    Parameters:
    df: DataFrame with daily data and engineered features
    date_col: name of date column
    security_col: name of security identifier column  
    target_col: name of binary target column
    """
    
    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create month-year identifier
    df['month_year'] = df[date_col].dt.to_period('M')
    
    # Get the last trading day of each month for each security
    monthly_data = df.groupby([security_col, 'month_year']).last().reset_index()
    
    # Sort by security and date
    monthly_data = monthly_data.sort_values([security_col, 'month_year'])
    
    # Create target for NEXT month
    monthly_data['target_next_month'] = monthly_data.groupby(security_col)[target_col].shift(-1)
    
    # Remove rows where we don't have next month's target
    monthly_data = monthly_data.dropna(subset=['target_next_month'])
    
    return monthly_data

def train_xgb_portfolio_model(df, feature_columns, date_col='date', security_col='security', target_col='include_exclude', test_size=0.2):
    """
    Train XGBoost model to predict next month's portfolio inclusion
    
    Parameters:
    df: DataFrame with features and target
    feature_columns: list of feature column names
    """
    
    print("Preparing data for next month prediction...")
    monthly_data = prepare_next_month_target(df, date_col, security_col, target_col)
    
    print(f"Monthly dataset shape: {monthly_data.shape}")
    print(f"Using {len(feature_columns)} features")
    
    # Check for missing values in features
    missing_features = [col for col in feature_columns if monthly_data[col].isnull().any()]
    if missing_features:
        print(f"Warning: Missing values found in features: {missing_features}")
        monthly_data = monthly_data.dropna(subset=feature_columns)
        print(f"Shape after removing missing values: {monthly_data.shape}")
    
    # Time-based split to avoid look-ahead bias
    monthly_data = monthly_data.sort_values('month_year')
    split_idx = int(len(monthly_data) * (1 - test_size))
    
    train_data = monthly_data.iloc[:split_idx]
    test_data = monthly_data.iloc[split_idx:]
    
    # Features and target
    X_train = train_data[feature_columns]
    y_train = train_data['target_next_month']
    X_test = test_data[feature_columns]
    y_test = test_data['target_next_month']
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Target distribution in training: {y_train.value_counts().to_dict()}")
    
    # XGBoost hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    
    # Base model
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Time series cross-validation for hyperparameter tuning
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("Starting hyperparameter tuning...")
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    test_auc = roc_auc_score(y_test, y_test_prob)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
        
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Cross-validation AUC: {grid_search.best_score_:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("\nTest Set Confusion Matrix:")
    print(cm)
    
    # --- Start of new code for accuracy dictionary ---
    
   # Calculate required metrics for the prediction results dictionary
    accuracy = accuracy_score(y_test, y_test_pred)
    precision_include = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    recall_include = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    f1_score_value = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    
    total_predictions = len(y_test)
    accurate_predictions = int(accuracy * total_predictions)
    

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {min(10, len(feature_columns))} Most Important Features:")
    print(feature_importance.head(10))
      # Create the prediction_results dictionary
   # Feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)      
    prediction_results = {
        'date' : pd.to_datetime(df[date_col]).max(),
        'total_predictions': total_predictions,
        'accurate_predictions': accurate_predictions,
        'accuracy_rate': accuracy,
        'precision_include': precision_include,
        'recall_include': recall_include,
        'f1_score': f1_score_value,
        'feature_importance': feature_importance_df.set_index('feature')['importance'].to_dict()
    }  
    #return best_model, feature_importance, monthly_data, y_test, y_test_prob 
    return best_model, feature_importance, monthly_data, y_test, y_test_prob, prediction_results

def plot_model_performance(y_true, y_prob, feature_importance_df, title="Portfolio Model Performance"):
    """
    Plot ROC curve and feature importance
    
    Parameters:
    y_true: true binary labels
    y_prob: predicted probabilities
    feature_importance_df: DataFrame with feature importance
    title: plot title
    """
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(12, 5))
    
    # ROC Curve subplot
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Feature importance subplot
    plt.subplot(1, 2, 2)
    top_features = feature_importance_df.head(10)
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return auc_score

def predict_next_month_portfolio(model, df, feature_columns, date_col='date', security_col='security'):
    """
    Generate predictions for the next month's portfolio
    """
    
    # Get the most recent month's data for each security
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy['month_year'] = df_copy[date_col].dt.to_period('M')
    
    # Get last trading day data for the most recent month
    latest_month_data = df_copy.groupby([security_col, 'month_year']).last().reset_index()
    current_month_data = latest_month_data[latest_month_data['month_year'] == latest_month_data['month_year'].max()]
    
    # Make predictions
    X_current = current_month_data[feature_columns]
    predictions = model.predict(X_current)
    probabilities = model.predict_proba(X_current)[:, 1]
    
    # Results dataframe
    results = pd.DataFrame({
        'security': current_month_data[security_col],
        'current_month': current_month_data['month_year'].astype(str),
        'predicted_include_next_month': predictions,
        'inclusion_probability': probabilities
    })
    
    results = results.sort_values('inclusion_probability', ascending=False).reset_index(drop=True)
    
    return results

def generate_monthly_portfolio_predictions(model, processed_data, feature_cols, 
                                         date_col='Date', security_col='Ticker'):
    """
    Generate monthly portfolio predictions and store in dictionary format
    """
    
    predicted_portfolio = {}
    
    # Get unique months from the data
    processed_data[date_col] = pd.to_datetime(processed_data[date_col])
    processed_data['month_year'] = processed_data[date_col].dt.to_period('M')
    
    # Get the most recent month's data for each security
    latest_month_data = processed_data.groupby([security_col, 'month_year']).last().reset_index()
    current_month_data = latest_month_data[latest_month_data['month_year'] == latest_month_data['month_year'].max()]
    
    # Make predictions
    X_current = current_month_data[feature_cols]
    predictions = model.predict(X_current)
    probabilities = model.predict_proba(X_current)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'security': current_month_data[security_col],
        'predicted_include_next_month': predictions,
        'inclusion_probability': probabilities
    })
    
    # Filter for securities to include
    include_securities = results[results['predicted_include_next_month'] == 1].copy()
    include_securities = include_securities.sort_values('inclusion_probability', ascending=False)
    
    # Get current date
    current_date = current_month_data['month_year'].iloc[0]
    
    # Store in dictionary format
    predicted_portfolio[current_date] = {
        'Ticker': include_securities['security'].tolist(),
        'Probability': include_securities['inclusion_probability'].tolist()
    }
    
    print(f"Portfolio for {current_date}:")
    print(f"  - Securities to include: {len(include_securities)}")
    print(f"  - Top 5 probabilities: {include_securities['inclusion_probability'].head().tolist()}")
    
    return predicted_portfolio, include_securities



# Example usage:
"""
# Define your feature columns
feature_columns = [
    'return', 
    'volatility', 
    '3_month_momentum', 
    '6_month_momentum'
    # Add any other engineered features you have
]

# Train the model
model, feature_importance, processed_data, y_test, y_test_prob = train_xgb_portfolio_model(
    df=your_dataframe,
    feature_columns=feature_columns,
    date_col='date',  # adjust to your date column name
    security_col='security',  # adjust to your security column name
    target_col='include_exclude'  # adjust to your target column name
)

# Plot model performance
auc_score = plot_model_performance(
    y_true=y_test,
    y_prob=y_test_prob,
    feature_importance_df=feature_importance,
    title="Portfolio Inclusion Prediction"
)

# Get next month predictions
next_month_predictions = predict_next_month_portfolio(
    model=model,
    df=your_dataframe,
    feature_columns=feature_columns,
    date_col='date',  # adjust to your date column name
    security_col='security'  # adjust to your security column name
)

print("Next Month Portfolio Predictions:")
print(next_month_predictions)

# To see which securities to include (probability > 0.5):
include_securities = next_month_predictions[next_month_predictions['predicted_include_next_month'] == 1]
print(f"\nSecurities to INCLUDE next month: {len(include_securities)}")
print(include_securities[['security', 'inclusion_probability']])
###############################################################################################################
# Usage:
predicted_portfolio, include_securities = generate_monthly_portfolio_predictions(
    model=model,
    processed_data=processed_data,
    feature_cols=feature_cols,
    date_col='Date',
    security_col='Ticker'
)

# Now you can access the data as you originally intended:
for date, portfolio in predicted_portfolio.items():
    print(f"\nDate: {date}")
    print(f"Tickers: {portfolio['Ticker'][:5]}...")  # First 5 tickers
    print(f"Probabilities: {portfolio['Probability'][:5]}...")  # First 5 probabilities

# Extract selected securities for optimization
selected_securities = include_securities['security'].tolist()

# Continue with your optimization workflow
returns_matrix = prepare_returns_matrix_safe(final_portfolio_data)
expected_returns, cov_matrix, securities = integrate_expected_returns_with_optimizer(
    returns_matrix=returns_matrix,
    selected_securities=selected_securities,
    expected_returns_method='multi_factor'
)



"""