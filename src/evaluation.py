"""
evaluation.py
Comprehensive evaluation metrics for regression model assessment.

Original metrics matching equations (1) to (5) of Munshi et al. (2026):
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error (%)
- R2: Coefficient of Determination

Extended metrics (added per professor's Review 1 feedback):
- Adjusted R2: Penalizes model complexity (accounts for number of features)
- Explained Variance Score: Proportion of variance explained by model
- Max Error: Worst-case single prediction error (MPa)
- Median Absolute Error: Robust central-tendency error metric
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, median_absolute_error
)

def compute_metrics(y_true, y_pred, n_features=5):
    """
    Compute all 9 regression metrics for given true and predicted values.
    Handles 1D or 2D arrays.
    
    Parameters:
        y_true: array of actual values
        y_pred: array of predicted values
        n_features: number of input features (default=5 for our FDM parameters)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    
    n = len(y_true)
    
    # Avoid zero division in MAPE
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # --- Extended metrics (Review 1 feedback) ---
    # Adjusted R²: penalizes for number of features
    if n > n_features + 1:
        adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - n_features - 1))
    else:
        adj_r2 = r2
    
    evs = explained_variance_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE_%': float(mape),
        'R2': float(r2),
        'Adjusted_R2': float(adj_r2),
        'Explained_Variance': float(evs),
        'Max_Error': float(max_err),
        'Median_AE': float(median_ae)
    }

def compute_normalised_errors(y_true, y_pred, y_min, y_max):
    """
    MSE / RMSE / MAE after min-max scaling the strengths to [0, 1] with the
    dataset-wide range. Used for a scale-matched comparison with Table 4 of the
    reference paper, whose (sub-1) error values appear to be reported on
    normalised targets.
    """
    span = float(y_max - y_min)
    t = (np.asarray(y_true, dtype=np.float64).ravel() - y_min) / span
    p = (np.asarray(y_pred, dtype=np.float64).ravel() - y_min) / span
    mse = mean_squared_error(t, p)
    return {
        'MSE_norm': float(mse),
        'RMSE_norm': float(np.sqrt(mse)),
        'MAE_norm': float(mean_absolute_error(t, p))
    }

def evaluate_multi_target(y_true, y_pred, target_names=['Tensile_Strength_Mpa', 'Compressive_Strength_Mpa'], n_features=5):
    """
    Compute metrics separately for each target column.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    results = {}
    for i, name in enumerate(target_names):
        results[name] = compute_metrics(y_true[:, i], y_pred[:, i], n_features)
        
    return results

def format_metrics_table(results_dict):
    """
    Converts a dictionary of model results into a clean pandas DataFrame.
    """
    records = []
    for model_name, targets in results_dict.items():
        for target_name, metrics in targets.items():
            record = {'Model': model_name, 'Target': target_name}
            record.update({k: round(v, 6) for k, v in metrics.items()})
            records.append(record)
    return pd.DataFrame(records)
