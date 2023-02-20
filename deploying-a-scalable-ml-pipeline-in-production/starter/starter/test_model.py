"""
Unit test for ml.model module

Author: quangdungluong
Date: February 20, 2023
"""
import os
import sys
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble._forest import BaseForest

sys.path.insert(0, os.path.dirname(__file__))

from ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    # Model is ClassifierMixin and BaseForest instance
    X = np.random.rand(5, 108)
    y = np.random.randint(1, size=5)
    model = train_model(X, y)
    assert isinstance(model, ClassifierMixin) and isinstance(model, BaseForest)

def test_compute_model_metrics():
    y = [1, 1, 1, 0]
    pred = [1, 1, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y, pred)
    # Expect precision=2/3, recall=2/3, fbeta=2/3
    assert precision == 2/3
    assert recall == 2/3
    assert fbeta == 2/3

def test_inference():
    X = np.random.rand(5, 108)
    y = np.random.randint(1, size=5)
    model = train_model(X, y)
    pred = inference(model, X)
    assert y.shape == pred.shape