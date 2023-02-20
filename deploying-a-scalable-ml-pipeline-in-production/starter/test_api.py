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
        'age': 52,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 209642,
        'education': 'HS-grad',
        'education_num': 9,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 45,
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