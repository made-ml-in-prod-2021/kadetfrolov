from typing import Union, Dict
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import pickle
import json
from entities.train_pipeline_params import TrainingPipelineParams

sklearn_model = Union[RandomForestClassifier, LogisticRegression, ColumnTransformer]


def train_model(dataset: pd.DataFrame, target: pd.Series, train_params: TrainingPipelineParams) -> sklearn_model:
    model_name = train_params.model
    if model_name == 'RandomForestClassifier':
        model = RandomForestClassifier()
    elif model_name == 'LogisticRegression':
        model = LogisticRegression()
    else:
        raise NotImplementedError
    model.fit(dataset, target)
    return model


def predict_model(pipeline: Pipeline, dataset: pd.DataFrame) -> np.ndarray:
    result = pipeline.predict(dataset)
    return result


def evaluate_model(predictions: np.ndarray, target: pd.Series) -> Dict[str, float]:
    result = {'accuracy': round(accuracy_score(target, predictions), 4),
              'roc_auc': round(roc_auc_score(target, predictions), 4),
              'f1': round(f1_score(target, predictions), 4),
              'precision': round(precision_score(target, predictions), 4),
              'recall': round(recall_score(target, predictions), 4)}
    return result


def create_inference_pipeline(model: sklearn_model, transformer: ColumnTransformer) -> Pipeline:
    # Pipeline of transforms with a final estimator.
    # Sequentially apply a list of transforms and a final estimator.
    # Intermediate steps of the pipeline must be ‘transforms’,
    # that is, they must implement fit and transform methods. The final estimator only needs to implement fit.
    pipeline = Pipeline([('feature_part', transformer), ('model_part', model)])
    return pipeline


def serialize_artefact(model: sklearn_model, save_path: str):
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)


def save_metrics(metrics: dict, save_path: str):
    with open(save_path, 'w') as file:
        json.dump(metrics, file)


if __name__ == "__main__":
    import data.make_dataset
    import entities.train_pipeline_params
    path_data = '../data/heart.csv'
    path_config = '../config/train_config_forest.yaml'
    path_model_save = '../models/model.pkl'
    path_model_metrics = '../models/metrics.json'

    dataset_main = data.make_dataset.read_data(path_data)
    params = entities.train_pipeline_params.read_training_pipeline_params(path_config)
    trainset, valset = data.make_dataset.split_train_val_data(dataset_main, params)
    x_train = trainset.drop(params.feature_params.target_feature, axis=1)
    y_train = trainset[params.feature_params.target_feature]
    x_val = valset.drop(params.feature_params.target_feature, axis=1)
    y_val = valset[params.feature_params.target_feature]
    model = train_model(x_train, y_train, params)

    pipeline = Pipeline([('only model', model)])
    preds = predict_model(pipeline, x_val)
    metrics = evaluate_model(preds, y_val)

    # print(metrics)
    serialize_artefact(model, path_model_save)
    save_metrics(metrics, path_model_metrics)


