import uvicorn
import pickle
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Load the trained model
with open("starter/models/rf_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Define the FastAPI app
app = FastAPI()

# Define the unit tests
def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the API!"}

def test_predict_1():
    with TestClient(app) as client:
        input_data = {"age": 25, "workclass": "Private", "education_level": "11th", "education_num": 7, "marital_status": "Never-married", "occupation": "Machine-op-inspct", "relationship": "Own-child", "race": "Black", "sex": "Male", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States"}
        response = client.post("/predict", json=input_data)
        assert response.status_code == 200
        assert response.json() == {"prediction": "<=50K"}

def test_predict_2():
    with TestClient(app) as client:
        input_data = {"age": 45, "workclass": "Self-emp-not-inc", "education_level": "Bachelors", "education_num": 13, "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial", "relationship": "Husband", "race": "White", "sex": "Male", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 50, "native_country": "United-States"}
        response = client.post("/predict", json=input_data)
        assert response.status_code == 200
        assert response.json() == {"prediction": ">50K"}

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
