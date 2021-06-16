import pickle
from fastapi import FastAPI, HTTPException
from typing import List
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
import pandas as pd


class ResponseItem(BaseModel):
    pred: int


class RequestItem(BaseModel):
    data: List[List]
    features: List[str]


def load_artefact(path: str):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def create_pipeline(path: str) -> Pipeline:
    model = load_artefact(f'{path}/model.pkl')
    transformer = load_artefact(f'{path}/transformer.pkl')
    pipeline = Pipeline([('transformer', transformer),
                         ('model', model)])
    return pipeline


def make_prediction(pipeline: Pipeline, data: pd.DataFrame) -> List[ResponseItem]:
    preds = pipeline.predict(data)
    return [ResponseItem(pred=p) for p in preds]


app = FastAPI()
pipeline = None
pipeline = create_pipeline('models')
true_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
                "thal"]


@app.get('/')
async def start():
    return 'Welcome to the main page!'


@app.on_event('startup')
async def load_model():
    global pipeline
    pipeline = create_pipeline('models')


@app.get('/predict', response_model=List[ResponseItem])
async def predict(item: RequestItem):
    if len(item.features) != len(item.data[0]):
        raise HTTPException(status_code=400, detail=f'Get {len(item.features)}-features while data has {len(item.data[0])}')
    if item.features != true_features:
        raise HTTPException(status_code=400, detail='Features mismatch')
    df = pd.DataFrame(item.data, columns=item.features)
    return make_prediction(pipeline, df)
