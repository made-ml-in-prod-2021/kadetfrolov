from data.make_dataset import read_data
from models.model_fit_predict import create_inference_pipeline, predict_model
from entities.train_pipeline_params import TrainingPipelineParams
from typing import Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import numpy as np
import click
import pickle
import sys
sklearn_models = Union[RandomForestClassifier, LogisticRegression]


def predict(model: sklearn_models, transformer: ColumnTransformer, data_path: str) -> np.ndarray:
    data = read_data(data_path).drop('target', axis=1)
    inference_pipeline = create_inference_pipeline(model, transformer)
    preds = predict_model(inference_pipeline, data)
    return preds


def predict_command(model_path: str, transformer_path: str,  input_path: str, output_path: str) -> np.ndarray:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    preds = predict(model, transformer, input_path)
    np.savetxt(output_path, preds, fmt="%d", comments='')


_, model_path, transformer_path, input_path, output_path = sys.argv
predict_command(model_path, transformer_path, input_path, output_path)

