# # Put the code for your API here.
import pickle
from pydantic import BaseModel

from fastapi import FastAPI

from starter.ml.model import inference

app = FastAPI()

# Define the Pydantic model for the input data
class InputData(BaseModel):
    age: int
    workclass: str
    education_level: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.get('/')
def index() -> str:
    return "Welcome to this application"


@app.post('/inference')
def salary_inference(X : InputData) -> str:
    preds = inference(X)
    return preds

# import uvicorn
# import pandas as pd
# import pickle
# from fastapi import FastAPI, TestClient
# from pydantic import BaseModel

# # Load the trained model


# # Define the FastAPI app
# app = FastAPI()



# # Define the root endpoint
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the API!"}

# # Define the prediction endpoint
# @app.post("/predict")
# async def predict(input_data: InputData):
#     # Convert the input data into a Pandas DataFrame
#     input_df = pd.DataFrame([input_data.dict()])

#     # Inference using pipe
#     pipe = inference(X)
#     # Perform the prediction using the trained model
#     prediction = pipe.predict(input_df)

#     # Return the prediction as a dictionary
#     return {"prediction": prediction.tolist()[0]}

