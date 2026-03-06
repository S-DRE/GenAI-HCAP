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

# Use 0.0.0.0 so Docker's port mapping (host:8000 → container:8000) works.
# The host-side binding is controlled by docker-compose ports (127.0.0.1:8000:8000).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
