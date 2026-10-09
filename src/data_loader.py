"""
data_loader.py
Module for loading, validating and splitting the FDM 3D-printing dataset of ABS
mechanical properties (Tensile and Compressive Strengths).

The dataset (383 samples) is the one released with Munshi et al. (2026),
Next Materials 10, 101420. According to Section 2.5 of the paper it was compiled
from peer-reviewed studies and test reports on ASTM/ISO standard specimens.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    'Nozzle_Temp_C',
    'Bed_Temp_C',
    'Print_Speed_mm_per_s',
    'Layer_Height_mm',
    'Infill_Density_percent'
]

TARGET_COLUMNS = [
    'Tensile_Strength_Mpa',
    'Compressive_Strength_Mpa'
]

FEATURE_DISPLAY_NAMES = [
    'Nozzle Temp (°C)',
    'Bed Temp (°C)',
    'Print Speed (mm/s)',
    'Layer Height (mm)',
    'Infill Density (%)'
]

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data',
    'Experimental_Dataset.xlsx'
)


def load_raw_data(filepath=DEFAULT_DATA_PATH):
    """
    Load the experimental dataset from the Excel file.
    The first 4 rows hold title/author metadata and are skipped.
    Only the 5 feature and 2 target columns are returned.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_excel(filepath, skiprows=4)
    for col in FEATURE_COLUMNS + TARGET_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in dataset.")

    df = df[FEATURE_COLUMNS + TARGET_COLUMNS].dropna().reset_index(drop=True)
    df = df.astype(float)
    return df


def get_param_bounds(df=None):
    """
    Operating window of each process parameter, taken directly from the data
    (identical to Table 2 of the reference paper for the levels present in the data):
    nozzle 200-250 °C, bed 50-110 °C, speed 10-70 mm/s, layer 0.2-0.8 mm, infill 20-100 %.
    """
    if df is None:
        df = load_raw_data()
    return {col: (float(df[col].min()), float(df[col].max())) for col in FEATURE_COLUMNS}


def get_param_levels(df=None):
    """Distinct experimental levels of each process parameter present in the dataset."""
    if df is None:
        df = load_raw_data()
    return {col: sorted(df[col].unique().tolist()) for col in FEATURE_COLUMNS}


def get_train_test_data(filepath=DEFAULT_DATA_PATH, test_size=0.2, random_state=42):
    """
    Hold-out split (80 % train / 20 % test) on raw, unscaled values.
    Scaling is performed inside each model pipeline (see models_common.wrap_model),
    so the scalers are always fitted on the training portion only.

    Returns:
        X_train, X_test, y_train, y_test, df
    """
    df = load_raw_data(filepath)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMNS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, df


if __name__ == '__main__':
    df = load_raw_data()
    print(f"Dataset successfully loaded. Shape: {df.shape}")
    print("\nParameter bounds:", get_param_bounds(df))
    print("Parameter levels:", get_param_levels(df))
    print("\nSummary Statistics:")
    print(df.describe().round(2).T)
