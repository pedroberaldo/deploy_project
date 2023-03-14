# # # Put the code for your API here.
# import pickle
# from pydantic import BaseModel

# from fastapi import FastAPI, Response, status

# from starter.starter.ml.model import inference
# from starter.starter.ml.data import process_data
# import pandas as pd

# app = FastAPI()

# # Define the Pydantic model for the input data
# class InputData(BaseModel):
#     age: int
#     workclass: str
#     education: str
#     education_num: int
#     marital_status: str
#     occupation: str
#     relationship: str
#     race: str
#     sex: str
#     capital_gain: int
#     capital_loss: int
#     hours_per_week: int
#     native_country: str
#     fnlwgt : int
#     class Config:
#         schema_extra = {
#             "example": {
#                 "workclass": "state_gov",
#                 "education": "11th",
#                 "marital_status": "Never_married",
#                 "occupation": "adm_clerical",
#                 "relationship": "not_in_family",
#                 "race": "white",
#                 "sex": "Male",
#                 "native_country": "Cuba",
#                 "age": 29,
#                 "fnlwgt": 77516, 
#                 "education_num": 13,
#                 "capital_gain": 2174,
#                 "capital_loss": 0,
#                 "hours_per_week": 40
#             }
#         }

# @app.get('/')
# def index() -> Response:
#     return Response(
#             status_code=status.HTTP_200_OK,
#             content="FastAPI prediction is up and running!",
#         )
# @app.post('/inference')
# def salary_inference(input_json : InputData) -> Response:
#     cat_features = [
#         "workclass",
#         "education",
#         "marital_status",
#         "occupation",
#         "relationship",
#         "race",
#         "sex",
#         "native_country",
#     ]
#     with open('starter/starter/models/encoder.pkl', 'rb') as file:
#         encoder = pickle.load(file)
#     with open('starter/starter/models/lb.pkl', 'rb') as file:
#         lb = pickle.load(file)
#     with open('starter/starter/models/rf_model.pkl', 'rb') as file:
#         model = pickle.load(file)

#     X = pd.DataFrame(dict(input_json), index=[0])
#     X,_,_,_ = process_data(X, categorical_features=cat_features, encoder=encoder,lb=lb, training=False)
#     preds = inference(model, X)
#     result = lb.inverse_transform(preds[0])[0]
#     response = Response(
#             status_code=status.HTTP_200_OK,
#             content="Predicted income: " + result,
#         )
#     return response


# # @app.post('/inference')
# # def salary_inference(input_json : InputData) -> str:
# #     cat_features = [
# #         "workclass",
# #         "education",
# #         "marital_status",
# #         "occupation",
# #         "relationship",
# #         "race",
# #         "sex",
# #         "native_country",
# #     ]
# #     with  open('starter/starter/models/encoder.pkl', 'rb') as file:
# #         encoder = pickle.load(file)
# #     with  open('starter/starter/models/lb.pkl', 'rb') as file:
# #         lb = pickle.load(file)
# #     with  open('starter/starter/models/rf_model.pkl', 'rb') as file:
# #         model = pickle.load(file)
# #     try:
# #     # your code that raises a ValidationError
# #         X = pd.DataFrame(dict(input_json), index=[0])
# #         X,_,_,_ = process_data(X, categorical_features=cat_features, encoder=encoder,lb=lb, training=False)
# #         preds = inference(model, X)
# #         return preds
# #     except ValidationError as e:
# #         print(e)
    
# # import uvicorn
# # import pandas as pd
# # import pickle
# # from fastapi import FastAPI, TestClient
# # from pydantic import BaseModel

# # # Load the trained model


# # # Define the FastAPI app
# # app = FastAPI()



# # # Define the root endpoint
# # @app.get("/")
# # async def root():
# #     return {"message": "Welcome to the API!"}

# # # Define the prediction endpoint
# # @app.post("/predict")
# # async def predict(input_data: InputData):
# #     # Convert the input data into a Pandas DataFrame
# #     input_df = pd.DataFrame([input_data.dict()])

# #     # Inference using pipe
# #     pipe = inference(X)
# #     # Perform the prediction using the trained model
# #     prediction = pipe.predict(input_df)

# #     # Return the prediction as a dictionary
# #     return {"prediction": prediction.tolist()[0]}

import pickle
from pydantic import BaseModel

from fastapi import FastAPI, Response, status

from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data
import pandas as pd

app = FastAPI()

# Define the Pydantic model for the input data
class InputData(BaseModel):
    age: int
    workclass: str
    education: str
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
    fnlwgt : int
    class Config:
        schema_extra = {
            "example": {
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
        }

@app.get('/')
def index() -> Response:
    return Response(
            status_code=status.HTTP_200_OK,
            content="FastAPI for Udacity course :)"
        )


@app.post('/inference')
def salary_inference(input_json : InputData) -> str:
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    with  open('starter/starter/models/encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with  open('starter/starter/models/lb.pkl', 'rb') as file:
        lb = pickle.load(file)
    with  open('starter/starter/models/rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    # X = pd.DataFrame(dict(input_json), index=[0])
    # X,_,_,_ = process_data(X, categorical_features=cat_features, encoder=encoder,lb=lb, training=False)
    # preds = inference(model, X)
    # return preds
    X = pd.DataFrame(dict(input_json), index=[0])
    X,_,_,_ = process_data(X, categorical_features=cat_features, encoder=encoder,lb=lb, training=False)
    preds = inference(model, X)
    result = lb.inverse_transform(preds[0])[0]
    response = Response(
            status_code=status.HTTP_200_OK,
            content="Predicted income: " + result,
        )
    return response