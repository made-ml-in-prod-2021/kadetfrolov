FROM python:3.7
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
EXPOSE 80
WORKDIR .

COPY models/model.pkl /models/model.pkl
COPY models/transformer.pkl /models/transformer.pkl
COPY app.py /app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]