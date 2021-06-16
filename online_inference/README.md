Репозиторий содержит обученную модель для решения задачи предсказания заболевания.
Данные для построения модели были взяты [здесь](https://www.kaggle.com/ronitf/heart-disease-uci).
Inference модели был обернут в REST сервис (был использован FastAPI)
Создадим локально conda-окружение для тестирования модулей:
* ```conda create -n .hw2_prod```
* ```conda activate .hw2_prod```
* ```conda install pip```
* ```pip install -r requirements.txt```

## Локально
Все команды выполняются внутри директории ```online_inference```.
Сервис поднимается командой (из окружения .hw2_prod):
* ```uvicorn app:app```

Для валидации метода ```/predict``` был подготовлен модуль ```make_app_request.py``` (прогон синтетических данных и trainset), который запускается (из окружения .hw2_prod):
* ```python make_app_request.py```

Реализована валидация поступающих данных. С примером валидации можно ознакомиться с помощью тестов в модуле ```test_app.py``` (из окружения .hw2_prod):
* ```pytest```

## Docker
Для использования модели в другой среде был создан dockerfile (возможно команду потребуется запустить с sudo):
* ```docker build -t nikovtb/online_inference .```

Запуск локально контейнера (возможно команду потребуется запустить с sudo):
* ```docker run -p 8000:80 nikovtb/online_inference```

Докер образ был опубликован [здесь](https://hub.docker.com/repository/docker/nikovtb/online_inference), использовать у себя можно с помощью:
* ```docker pull nikovtb/online_inference:latest```
* ```docker run -p 8000:80 nikovtb/online_inference:latest```

Как и прежде, развернутый в докере сервис можно тестировать у себя локально (не забудьте зайти в окружение .hw2_prod):
* ```python make_app_request.py```

## В заключении
Вторая домашняя работа выполнена в рамках hard deadline, все пункты за исключением №6 (оптимизация докер образа) были выполнены
Суммарное число баллов: (3 + 3 + 2 + 3 + 4 + 0 + 2 + 1 + 1) * 0.65 = 12.35

P.S. Буду благодарен за советы по оптимизации моего dockerfile.
