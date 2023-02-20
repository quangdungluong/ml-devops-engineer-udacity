"""
A function that computes model metrics on slices of the data.

Author: quangdungluong
Date: February 20, 2023
"""
import os
import pickle
import sys

import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference


def metrics(data, gt, pred, feature, categorical_features, save_path):
    with open(save_path, 'a') as f:
        sys.stdout = f
        print("Model Performance on slices with fixed value is:", feature)
        
        for value in data[feature].unique():
            value_index = data.index[data[feature] == value]
            print(f"Feature {feature} = {value}")
            precision, recall, fbeta = compute_model_metrics(gt[value_index], pred[value_index])
            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, fbeta: {fbeta:.3f}")
            print('-------------------------------------------------')
        print("#"*10)

if __name__ == "__main__":
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
    dir = os.path.dirname(__file__)
    test_data = pd.read_csv(os.path.join(dir, "../data/test_df.csv"))
    model = pickle.load(open(os.path.join(dir, "../model/rf_model.pkl"), 'rb'))
    encoder = pickle.load(open(os.path.join(dir, "../model/encoder.pkl"), 'rb'))
    lb = pickle.load(open(os.path.join(dir, "../model/lb.pkl"), 'rb'))

    X_test, y_test, _, _ = process_data(test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    predictions = inference(model, X_test)

    save_path = os.path.join(dir, "slice_output.txt")

    for feature in cat_features:
        metrics(test_data, y_test, predictions, feature, cat_features, save_path)