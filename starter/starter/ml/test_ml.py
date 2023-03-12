import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

# os.chdir("ml/")
# from model import train_model, compute_model_metrics, inference
import os
import pickle
import sys

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Add parent directory to Python path
from ml.model import *
from ml.data import *

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

@pytest.fixture(scope="session")
def get_data():
    data = pd.read_csv('cleaned_census.csv')
    train, test = train_test_split(data, test_size=0.20)
    return {
        'train': train, 
        'test' : test
    }

@pytest.fixture(scope="session")
def get_process_data_train(get_data):
    X_train, y_train, encoder, lb = process_data(
        get_data['train'], categorical_features=cat_features, label="salary", training=True
    )
    return {
        'X_train' : X_train, 
        'y_train' : y_train,
        'encoder' : encoder,
        'lb' :  lb
        }

@pytest.fixture(scope="session")
def get_process_data_test(get_data, get_process_data_train):
    X_test, y_test, encoder, lb = process_data(
        get_data['test'], categorical_features=cat_features, label="salary", training=False, encoder=get_process_data_train['encoder'], lb=get_process_data_train['lb']
    )
    return {
        'X_test': X_test,
        'y_test': y_test
    }

def test_train_model():
    # Test that the train_model function returns a trained model
    # model = train_model(X_train, y_train)
    model = pickle.load(open("starter/starter/models/rf_model.pkl",'rb'))
    assert model is not None

def test_compute_model_metrics():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    
    metrics = compute_model_metrics(y_true, y_pred)
    
    assert isinstance(metrics[0], float) # precsion
    assert isinstance(metrics[1], float) # recall
    assert isinstance(metrics[2], float) # f1

def test_inference(get_process_data_test):
    # Test that inference returns expected results for a given input
    # model = train_model(X_train, y_train)
    model = pickle.load(open("starter/starter/models/rf_model.pkl",'rb'))
    preds = inference(model, get_process_data_test['X_test'])
    assert len(preds) == len(get_process_data_test['y_test'])

def test_process_data_training_mode(get_data, get_process_data_train):
    # Test training mode
    X_processed, y_processed, encoder, lb = process_data(get_data['train'], categorical_features=cat_features, label="salary", training=True)
    assert X_processed.shape == (26048, 108)
    assert y_processed.shape == (26048,)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

def test_process_data_inference_mode(get_data, get_process_data_train):
    # Test inference mode
    X_processed, y_processed, get_process_data_train['encoder'], get_process_data_train['lb'] = process_data(get_data['test'], categorical_features=cat_features, label="salary", training=False, encoder=get_process_data_train['encoder'], lb=get_process_data_train['lb'])
    assert X_processed.shape == (6513, 108)
    assert y_processed.shape == (6513,)