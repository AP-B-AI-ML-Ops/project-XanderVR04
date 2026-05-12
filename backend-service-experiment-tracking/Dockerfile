FROM python:3.13-slim

RUN pip install -U pip
RUN pip install psycopg2-binary mlflow uvicorn

ENV DB_USER=admin
ENV DB_PASSWORD=postgres

CMD ["sh", "-c", "mlflow server \
     --host 0.0.0.0 \
     --port 5000 \
     --backend-store-uri postgresql+psycopg2://${DB_USER}:${DB_PASSWORD}@database/mlflow_db \
     --default-artifact-root /mlflow/artifacts \
     --allowed-hosts '*'"]
