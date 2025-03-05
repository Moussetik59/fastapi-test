import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest
from fastapi.testclient import TestClient
from main import app  

client = TestClient(app)

def test_predict_positive():
    """Test if the API returns a positive prediction"""
    response = client.post("/predict", json={"text": "I love this product, it's amazing!"})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["positive", "negative"]

def test_predict_negative():
    """Test if the API returns a negative prediction"""
    response = client.post("/predict", json={"text": "This service is terrible, I hate it!"})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["positive", "negative"]

def test_feedback():
    """Test sending feedback to the API"""
    feedback_data = {
        "text": "Example tweet",
        "prediction": "positive",
        "validation": False
    }
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    assert response.json()["message"] == "Feedback received, thank you!"
