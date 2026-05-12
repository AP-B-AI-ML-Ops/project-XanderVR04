FROM python:3.11-slim

RUN pip install -U pip

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

RUN prefect config set PREFECT_API_URL=http://orchestration:4200/api

RUN chmod +x /app/startPoolWorkers.sh

CMD ["/app/startPoolWorkers.sh"]
