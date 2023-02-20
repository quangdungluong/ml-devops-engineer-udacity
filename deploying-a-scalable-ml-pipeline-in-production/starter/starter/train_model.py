"""
Script to train machine learning model.

Author: quangdungluong
Date: February 20, 2023
"""

import os
import pickle
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from ml.data import process_data, remove_spaces
from ml.model import compute_model_metrics, inference, train_model

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
# Remove All spaces
data = remove_spaces(data)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)
# Save train_df and test_df to csv
train.to_csv("../data/train_df.csv", index=False)
test.to_csv("../data/test_df.csv", index=False)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Get the shape of input and output
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Train and save a model.
rf_model = train_model(X_train, y_train)

model_save_path = "../model/rf_model.pkl"
pickle.dump(rf_model, open(model_save_path, 'wb'))

encoder_save_path = "../model/encoder.pkl"
pickle.dump(encoder, open(encoder_save_path, 'wb'))

lb_save_path = "../model/lb.pkl"
pickle.dump(lb, open(lb_save_path, 'wb'))

# Get predictions
predictions = inference(rf_model, X_test)

# Compute metrics
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print(f"precision: {precision:.3f}, recall: {recall:.3f}, fbeta: {fbeta:.3f}")