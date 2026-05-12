#!/bin/bash

# Create a process work pool and start a worker in the background
prefect work-pool create --type process batch --overwrite
prefect worker start -p batch &

# Start the batch flow server (registers deployment + serves)
python /app/batch_flow.py
