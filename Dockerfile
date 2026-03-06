FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user — reduces blast radius if the container is ever compromised
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

# Bind to 127.0.0.1 by default so the port is not exposed on all interfaces.
# Override with --host 0.0.0.0 explicitly if deploying behind a reverse proxy.
CMD ["uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"]
