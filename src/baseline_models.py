"""
baseline_models.py
Replication of the reference models of Munshi et al. (2026), Next Materials 10, 101420:

1. Adam-optimised ANN (Section 2.6.1): feed-forward network, two tanh hidden layers,
   Adam optimiser, learning rate 1e-3, batch size 32, 1000 epochs, MSE loss.
   The paper does not state the layer width; 20 neurons per layer is used
   (same width as the paper's Bayesian network).
2. Bayesian-regularised ANN (Section 2.6.2): two hidden layers of 20 tanh neurons and
   a linear output layer. MATLAB's 'trainbr' (Levenberg-Marquardt with Bayesian
   regularisation) is not available in scikit-learn, so it is approximated by a
   second-order quasi-Newton optimiser (L-BFGS) with an L2 weight penalty (alpha).
3. Classical comparison models used in the paper's literature tables
   (Tables 5-7): SVR, Decision Tree and AdaBoost.

All models are wrapped with min-max scaling of inputs and targets (see model_utils).
"""

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

from model_utils import wrap_model


def get_baseline_models():
    """
    Returns configured (unfitted) baseline models replicating Munshi et al.
    """
    return {
        'Adam_ANN_MLP': wrap_model(MLPRegressor(
            hidden_layer_sizes=(20, 20),
            activation='tanh',
            solver='adam',
            learning_rate_init=1e-3,
            batch_size=32,
            max_iter=1000,
            random_state=42
        )),
        'Bayesian_Regularized_ANN': wrap_model(MLPRegressor(
            hidden_layer_sizes=(20, 20),
            activation='tanh',
            solver='lbfgs',
            alpha=0.01,
            max_iter=1000,
            random_state=42
        )),
        'Support_Vector_Regression_SVR': wrap_model(MultiOutputRegressor(
            SVR(C=20.0, epsilon=0.05, kernel='rbf')
        )),
        'Decision_Tree': wrap_model(DecisionTreeRegressor(
            max_depth=7, min_samples_split=4, random_state=42
        )),
        'AdaBoost': wrap_model(MultiOutputRegressor(
            AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        ))
    }
