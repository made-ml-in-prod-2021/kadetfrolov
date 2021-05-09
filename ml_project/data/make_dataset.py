import pandas as pd
from entities.split_params import SplittingParams
from typing import Tuple
from sklearn.model_selection import train_test_split


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(data: pd.DataFrame, split_params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    random_state = SplittingParams.random_state
    val_size = SplittingParams.val_size
    train_set, val_set = train_test_split(data, random_state=random_state, test_size=val_size)
    return train_set, val_set


if __name__ == '__main__':
    # TODO разобраться с относительными импортами в "from .split_params import SplittingParams"
    # файла train_pipeline_params
    from entities.train_pipeline_params import read_training_pipeline_params
    params = read_training_pipeline_params('../config/train_config_forest.yaml')
    path = 'heart.csv'
    data = read_data(path)
    train, val = split_train_val_data(data, params.splitting_params)
    print(train.shape)