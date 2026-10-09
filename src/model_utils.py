"""
model_utils.py
Helpers shared by every model in the project.

Every estimator is wrapped so that it takes raw process parameters and returns
strengths in MPa:

    TransformedTargetRegressor(
        regressor   = Pipeline([('scaler', MinMaxScaler()), ('model', estimator)]),
        transformer = MinMaxScaler()            # min-max scaling of the 2 targets
    )

This mirrors the min-max normalisation of Section 2.5.2 of Munshi et al. while
guaranteeing that the scalers are re-fitted on the training data of every split
(no leakage) and that downstream code (plots, SHAP, optimisation) never has to
handle scalers by hand.
"""

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def wrap_model(estimator):
    """Wrap an estimator with input and target min-max scaling."""
    return TransformedTargetRegressor(
        regressor=Pipeline([('scaler', MinMaxScaler()), ('model', estimator)]),
        transformer=MinMaxScaler(),
        check_inverse=False
    )


def fresh_copy(model):
    """Unfitted copy of a model (used so every split trains an independent model)."""
    return clone(model)


def unwrap_model(fitted_model):
    """
    Return (x_scaler, inner_estimator, y_scaler) of a fitted wrapped model.
    """
    pipe = fitted_model.regressor_
    return pipe.named_steps['scaler'], pipe.named_steps['model'], fitted_model.transformer_


def predict_mpa(model, X):
    """Predict both strengths (MPa) for raw parameter array X of shape (n, 5)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    pred = model.predict(X)
    return np.asarray(pred).reshape(len(X), -1)
