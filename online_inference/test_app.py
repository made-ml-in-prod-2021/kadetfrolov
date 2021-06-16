from app import app
from fastapi.testclient import TestClient

client = TestClient(app)
address = 'http://127.0.0.1:8000'


def test_prediction_right():
    json_data = {
        "data": [[58, 1, 1, 120, 284, 0, 0, 160, 0, 1.8, 1, 0, 2]],
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                     "ca", "thal"]
    }
    response = client.get(f'{address}/predict', json=json_data)
    assert response.status_code == 200
    assert response.json() == [{'pred': 0}]


# Given feature/column names do not match the ones for the data given during fit. This will fail from v0.24.
def test_prediction_shuffle_columns():
    json_data = {
        "data": [[1, 58, 1, 120, 284, 0, 0, 160, 0, 1.8, 1, 0, 2]],
        "features": ["sex", "age",  "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                     "ca", "thal"]
    }
    response = client.get(f'{address}/predict', json=json_data)
    assert response.status_code == 400
    assert response.json() == {'detail': 'Features mismatch'}


def test_prediction_features_data_mismatch():
    json_data = {
        "data": [[58, 1, 1, 120, 284, 0, 0, 160, 0, 1.8, 1, 0,1,1]],
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                     "ca"]
    }
    response = client.get(f'{address}/predict', json=json_data)
    assert response.status_code == 400
    assert response.json() == {'detail': 'Get 12-features while data has 14'}


def test_prediction_features_mismatch():
    json_data = {
        "data": [[58, 1, 1, 120, 284, 0, 0, 160, 0, 1.8, 1, 0,1]],
        "features": ["age", "s1x", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope",
                     "ca", "thal"]
    }
    response = client.get(f'{address}/predict', json=json_data)
    assert response.status_code == 400
    assert response.json() == {'detail': 'Features mismatch'}


