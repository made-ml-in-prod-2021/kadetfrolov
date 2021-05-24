from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from entities.train_pipeline_params import TrainingPipelineParams, FeatureParams
import numpy as np
import pandas as pd


def numerical_pipeline() -> Pipeline:
    pipeline = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                         ('normalize', StandardScaler())])
    return pipeline


def categorical_pipeline() -> Pipeline:
    pipeline = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))])
    return pipeline


def build_transformer(features: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer([('numerical_part', numerical_pipeline(), features.numerical_features),
                                     ('categorical_part', categorical_pipeline(), features.categorical_features)])
    return transformer


def make_features(data: pd.DataFrame, transformer: ColumnTransformer) -> pd.DataFrame:
    return transformer.transform(data)


def extract_target(data: pd.DataFrame, target: FeatureParams) -> pd.Series:
    return data[target.target_feature]


if __name__ == '__main__':
    from data.make_dataset import read_data
    from entities.train_pipeline_params import read_training_pipeline_params
    path_data = '../data/heart_nan.csv'
    path_config = '../config/train_config_forest.yaml'
    params = read_training_pipeline_params(path_config)
    dataset = read_data(path_data).drop(params.feature_params.target_feature, axis=1)
    transformer = build_transformer(params.feature_params)
    transformer.fit(dataset)
    data_transformed = make_features(dataset, transformer)
    assert data_transformed.shape == dataset.shape
