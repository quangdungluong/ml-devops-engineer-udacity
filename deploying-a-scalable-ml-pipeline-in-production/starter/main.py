"""
Implement RestAPI

Author: quangdungluong
Date: February 20, 2023
"""
import os
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference

dir = os.path.dirname(__file__)
app = FastAPI()
model = pickle.load(open(os.path.join(dir, "./model/rf_model.pkl"), 'rb'))
encoder = pickle.load(open(os.path.join(dir, "./model/encoder.pkl"), 'rb'))
lb = pickle.load(open(os.path.join(dir, "./model/lb.pkl"), 'rb'))

class SampleData(BaseModel):
    age: int = Field(None, example=50)
    workclass: str = Field(None, example="Self-emp-not-inc")
    fnlgt: int = Field(None, example=83311)
    education: str = Field(None, example="Bachelors")
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example="Married-civ-spouse")
    occupation: str = Field(None, example="Exec-managerial")
    relationship: str = Field(None, example="Husband")
    race: str = Field(None, example="White")
    sex: str = Field(None, example="Male")
    capital_gain: int = Field(None, example=0)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=13)
    native_country: str = Field(None, example="United-States")

@app.get("/")
async def welcome():
    return "Welcome to the homepage"

@app.post("/")
async def model_inference(sample_data: SampleData):
    data = {key.replace('_', '-'): [value] for key, value in sample_data.__dict__.items()}
    data = pd.DataFrame.from_dict(data)
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
    X, _, _, _ = process_data(data, cat_features, label=None, training=False, encoder=encoder, lb=lb)
    prediction = inference(model, X)
    print(prediction)
    return lb.inverse_transform(prediction)[0]