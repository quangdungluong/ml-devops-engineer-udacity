"""
POSTS to the API

Author: quangdungluong
Date: February 20, 2023
"""
import json

import requests

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

request = requests.post("https://udacity-project-3.onrender.com", data=json.dumps(data))
print(f"Status code:", request.status_code)
print(f"Inference result:", request.json())