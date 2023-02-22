"""
API

Author: quangdungluong
Date: February 22, 2023
"""
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from scoring import score_model
from diagnostics import dataframe_summary, missing_data, execution_time, outdated_packages_list, model_predictions
# import predict_exited_from_saved_model
import json
import os



app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'], config['model_name'])
test_data_path = os.path.join(config['test_data_path'], config['test_data'])
test_data = pd.read_csv(test_data_path)
model = pickle.load(open(output_model_path, 'rb'))


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    data_path = request.form.get('data_path')
    print(data_path)
    predictions = model_predictions(data_path=data_path)
    return json.dumps(predictions)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    f1_score = score_model(model, test_data)
    return json.dumps(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    stats = dataframe_summary()
    return json.dumps(stats.to_dict())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    diagnostic = {"execution_time": execution_time(),
               "na_percent": missing_data(),
               "dependencies": outdated_packages_list().to_dict('record')}
    return json.dumps(diagnostic)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
