import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ml import train_model, compute_model_metrics, inference

# Create a random dataset for testing
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def test_train_model():
    # Test that the train_model function returns a trained model
    model = train_model(X_train, y_train)
    assert model is not None

def test_compute_model_metrics():
    # Test that compute_model_metrics returns expected results for a given input
    y_true = np.array([1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == pytest.approx(0.5, 0.01)
    assert recall == pytest.approx(0.6, 0.01)
    assert fbeta == pytest.approx(0.55, 0.01)

def test_inference():
    # Test that inference returns expected results for a given input
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert len(preds) == len(y_test)

def test_compute_model_metrics_zero_division():
    # Test that compute_model_metrics returns 1 when division by zero occurs
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == 1
    assert recall == 1
    assert fbeta == 1

def test_train_model_param_grid():
    # Test that the param_grid dictionary has the expected keys
    keys = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
    param_grid = train_model(X_train, y_train).get_params()['param_distributions']
    assert all(k in param_grid.keys() for k in keys)
