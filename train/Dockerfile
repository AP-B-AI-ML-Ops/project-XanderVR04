FROM python:3.11-slim

RUN pip install -U pip

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "train.py"]
