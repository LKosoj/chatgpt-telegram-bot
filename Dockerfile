FROM python:3.12-slim

ENV PYTHONFAULTHANDLER=1 \
     PYTHONUNBUFFERED=1 \
     PYTHONDONTWRITEBYTECODE=1 \
     PIP_DISABLE_PIP_VERSION_CHECK=on \
     HOME=/home/bot \
     DB_PATH=/app/data/user_data.db \
     PLUGIN_STORAGE_ROOT=/app/data \
     SESSION_LOG_DIR=/app/log \
     BOT_DATA_DIR=/app/data \
     BOT_OUTPUT_DIR=/app/output \
     BOT_PLOTS_DIR=/app/plots

WORKDIR /app
COPY requirements.txt .
RUN apt-get update \
     && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg default-jre-headless graphviz g++ libc6-dev \
     && pip install -r requirements.txt --no-cache-dir \
     && apt-get purge -y --auto-remove g++ libc6-dev \
     && rm -rf /var/lib/apt/lists/* \
     && groupadd --system --gid 10001 bot \
     && useradd --system --uid 10001 --gid bot --create-home --home-dir /home/bot bot
COPY . .
RUN mkdir -p \
          data \
          output/plots \
          plots \
          usage_logs \
          log \
          bot/temp \
          uploads/webshot \
          bot/plugins/temp_pdfs \
          bot/plugins/pdf_cache \
          bot/plugins/document_metadata \
          bot/plugins/analytics \
          bot/plugins/language_data \
     && chown -R bot:bot \
          data \
          output \
          plots \
          usage_logs \
          log \
          bot/temp \
          uploads \
          bot/plugins/temp_pdfs \
          bot/plugins/pdf_cache \
          bot/plugins/document_metadata \
          bot/plugins/analytics \
          bot/plugins/language_data

USER bot
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
     CMD ["python", "-c", "import os; from pathlib import Path; cmd=Path('/proc/1/cmdline').read_bytes().replace(b'\\0', b' '); paths=('/app/data','/app/output','/app/plots','/app/usage_logs','/app/log','/app/bot/temp','/app/uploads','/tmp'); raise SystemExit(0 if b'python' in cmd and b'bot' in cmd and all(os.access(p, os.W_OK) for p in paths) else 1)"]

CMD ["python", "-m", "bot"]
