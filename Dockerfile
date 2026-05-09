FROM python:3.12-slim

ENV PYTHONFAULTHANDLER=1 \
     PYTHONUNBUFFERED=1 \
     PYTHONDONTWRITEBYTECODE=1 \
     PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app
COPY requirements.txt .
RUN apt-get update \
     && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg g++ libc6-dev \
     && pip install -r requirements.txt --no-cache-dir \
     && apt-get purge -y --auto-remove g++ libc6-dev \
     && rm -rf /var/lib/apt/lists/*
COPY . .

CMD ["python", "-m", "bot"]
