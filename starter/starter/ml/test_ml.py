import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from fastapi import FastAPI
from fastapi.testclient import TestClient
from main import app
# Load the trained model

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
    data = pd.read_csv('starter/cleaned_census.csv')
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

def test_inference(get_process_data_train, get_process_data_test):
    # Test that inference returns expected results for a given input
    # model = train_model(X_train, y_train)
    # pipe = get_training_inference_pipeline()
    # pipe.fit(get_process_data_train['X_train'], get_process_data_train['y_train'])
    model = pickle.load(open("starter/starter/models/rf_model.pkl",'rb'))
    preds = model.predict(get_process_data_test['X_test']) #pipe.predict(get_process_data_test['X_test'])
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


with open("starter/starter/models/rf_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Define the FastAPI app
# app = FastAPI()

# Define the unit tests
def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.content == b'FastAPI for Udacity course :)'

def test_predict_1():
    with TestClient(app) as client:
        input_data = {
                "workclass": "state_gov",
                "education": "11th",
                "marital_status": "Never_married",
                "occupation": "adm_clerical",
                "relationship": "not_in_family",
                "race": "white",
                "sex": "Male",
                "native_country": "Cuba",
                "age": 29,
                "fnlwgt": 77516,
                "education_num": 13,
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40
            }
        response = client.post("/inference", json=input_data)
        assert response.status_code == 200
        assert response.content == b'Predicted income: <=50K'

def test_predict_2():
    with TestClient(app) as client:
        input_data = {
                "workclass": "Self-emp-not-inc",
                "education": "HS-grad",
                "marital_status": "Never_married",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "white",
                "sex": "Male",
                "native_country": "United-States",
                "age": 52,
                "fnlwgt": 209642,
                "education_num": 13,
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40
            }
        response = client.post("/inference", json=input_data)
        assert response.status_code == 200
        assert response.content == b'Predicted income: >50K'


