import pickle
from fastapi import FastAPI
from fastapi.testclient import TestClient
from main import app
# Load the trained model
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
                "capital_loss": 0,#52,Self-emp-not-inc,209642,HS-grad,9,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,45,United-States,>50K
                "hours_per_week": 40
            }
        response = client.post("/inference", json=input_data)
        assert response.status_code == 200
        assert response.content == b'Predicted income: >50K'


