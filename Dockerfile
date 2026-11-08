FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY config.yaml .
COPY presets/ ./presets/
COPY src/ ./src/
# The index lives in the chroma service and chunk files are mounted (docker-compose.yml);
# ground_truth/ and eval/ are deliberately not in the image (see .dockerignore).

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
