Репозиторий с решением задачи предсказания наличия заболевания.

Данные для построения модели были взяты [здесь](https://www.kaggle.com/ronitf/heart-disease-uci).

Команды необходимо запускать из папки ml_project.

Настройка окружения:
* ```conda create -n .prod_hw1 python=3.7.10```
* ```conda activate .prod_hw1```
* ```conda install pip```
* ```pip3 install -r requirements.txt```

Запуск пайплайна случайного леса:
```python train_pipeline.py config/train_config_forest.yaml```

Запуск пайплайна линейной модели:
```python train_pipeline.py config/train_config_linear.yaml```

Запуск предсказания модели:
```python predict.py "models/model.pkl" "models/transformer.pkl" "data/heart.csv" "out.csv"```

После пробного прогона удалите окружение:
* ```conda env remove --name .prod_hw1```
