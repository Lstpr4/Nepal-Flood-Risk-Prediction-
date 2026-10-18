FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV KAGGLEHUB_CACHE=/app/.kagglehub

RUN mkdir -p /app/results /app/artifacts

# Default: run full training pipeline (override in compose)
CMD ["python", "run.py"]
