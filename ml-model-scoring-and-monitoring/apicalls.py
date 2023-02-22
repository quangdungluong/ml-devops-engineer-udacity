"""
Call API

Author: quangdungluong
Date: February 22, 2023
"""
import requests
import os
import json

config_path = "config.json"
config = json.load(open(config_path, 'r'))

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

#Call each API endpoint and store the responses
data_path = "./testdata/testdata.csv"
response1 = requests.post(f"{URL}/prediction", data={'data_path': data_path})
response2 = requests.get(f"{URL}/scoring")
response3 = requests.get(f"{URL}/summarystats")
response4 = requests.get(f"{URL}/diagnostics")

# combine all API responses
responses = {
    "response1": response1.json(),
    "response2": response2.json(),
    "response3": response3.json(),
    "response4": response4.json()
}

# write the responses to your workspace
with open(os.path.join(config['output_model_path'], config['api_return']), 'w') as f:
    json.dump(responses, f, indent=4)


