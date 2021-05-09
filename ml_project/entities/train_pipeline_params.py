from dataclasses import dataclass
from .split_params import SplittingParams
from .feature_params import FeatureParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    model: str


TrainingPipelineParams_schema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParams_schema()
        return schema.load(yaml.safe_load(input_stream))


if __name__ == "__main__":
    training_params = read_training_pipeline_params('../config/train_config_forest.yaml')
    print(training_params)
