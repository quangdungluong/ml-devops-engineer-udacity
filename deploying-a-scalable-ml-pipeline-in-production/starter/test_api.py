"""
Test the API

Author: quangdungluong
Date: February 20, 2023
"""
import json

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_welcome():
    request = client.get("/")
    assert request.status_code == 200
    assert request.json() == "Welcome to the homepage"

def test_inference_greater_50k():
    data = {
        'age': 33,
        'workclass': 'Local-gov',
        'fnlgt': 198183,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Prof-specialty',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Female',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': 'United-States'
    }
    request = client.post("/", data=json.dumps(data))
    assert request.status_code == 200
    assert request.json() == " >50K"

def test_inference_smaller_50k():
    data = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }
    request = client.post("/", data=json.dumps(data))
    assert request.status_code == 200
    assert request.json() == " <=50K"