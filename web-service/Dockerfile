FROM python:3.12-slim

RUN pip install -U pip

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "predict.py"]
