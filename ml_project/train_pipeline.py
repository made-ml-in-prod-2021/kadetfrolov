import logging
import sys
import click
from entities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from data.make_dataset import read_data, split_train_val_data
from features.build_features import extract_target, make_features, build_transformer
from models.model_fit_predict import *


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def run_train_pipeline(params: TrainingPipelineParams):
    logger.info(f"Start training with params: {params}")

    data = read_data(params.input_data_path)
    logger.info(f"Data shape is {data.shape}")

    train_set, val_set = split_train_val_data(data, params.splitting_params)

    y_train = extract_target(train_set, params.feature_params)
    y_val = extract_target(val_set, params.feature_params)
    x_train = train_set.drop(params.feature_params.target_feature, axis=1)
    x_val = val_set.drop(params.feature_params.target_feature, axis=1)
    logger.info(f"Training set shape: {x_train.shape}, validation set shape: {x_val.shape}")

    transformer = build_transformer(params.feature_params)
    transformer.fit(x_train)

    x_train_transformed = make_features(x_train, transformer)
    logger.info(f"Training set shape after preprocessing: {x_train_transformed.shape}")

    model = train_model(x_train_transformed, y_train, params)

    inference_pipeline = create_inference_pipeline(model, transformer)
    preds = predict_model(inference_pipeline, x_val)
    metrics = evaluate_model(preds, y_val)

    logger.info(f'Scores: {metrics}')
    serialize_model(model, params.output_model_path)
    save_metrics(metrics, params.metric_path)

    return params.output_model_path, metrics


@click.command(name='run_train_pipeline')
@click.argument('config_path')
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    run_train_pipeline(params)


if __name__ == '__main__':
    train_pipeline_command()
