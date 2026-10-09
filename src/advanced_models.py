"""
advanced_models.py
Proposed ensemble models (not evaluated in Munshi et al.):
1. Tuned Random Forest
2. Extra Trees
3. Histogram-based Gradient Boosting (one model per target)
4. XGBoost (one model per target)

All models are wrapped with min-max scaling of inputs and targets (see model_utils).
"""

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputRegressor

from model_utils import wrap_model

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def get_advanced_models():
    """
    Returns a dictionary of (unfitted) advanced ensemble models.
    """
    models = {
        'Random_Forest_Tuned': wrap_model(RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42
        )),
        'Extra_Trees_Ensemble': wrap_model(ExtraTreesRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_split=2,
            random_state=42
        )),
        'Hist_Gradient_Boosting': wrap_model(MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_depth=6,
                l2_regularization=0.1,
                random_state=42
            )
        ))
    }

    if XGB_AVAILABLE:
        models['XGBoost_Ensemble'] = wrap_model(MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=350,
                learning_rate=0.04,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.0,
                random_state=42
            )
        ))

    return models
