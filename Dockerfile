FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY config.yaml .
COPY src/ ./src/
# chroma_db/ is mounted at runtime (see docker-compose.yml), not baked in: it is gitignored,
# so COPY failed on a clean clone, and a mounted index can be swapped or corrupted from outside.

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
