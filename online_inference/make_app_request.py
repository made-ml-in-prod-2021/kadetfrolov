import pandas as pd
import requests
import numpy as np

if __name__ == '__main__':
    features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
                "thal"]
    num_requests = 5
    address = 'http://127.0.0.1:8000/predict'
    for i in range(num_requests):
        num_rows = np.random.randint(low=1, high=10, size=1).item()
        data = np.random.randint(low=1, high=190, size=(num_rows, len(features)))
        response = requests.get(address, json={'data': data.tolist(), 'features': features})
        print(response.json())

    csv_path = 'data/heart.csv'
    df = pd.read_csv(csv_path)
    for i in range(df.shape[0]):
        row = df.iloc[i].tolist()[:-1]
        response = requests.get(address, json={'data': [row], 'features': features})
        print(response.json())
