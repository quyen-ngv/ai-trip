FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the local Whisper model into the image; otherwise every new container downloads it during
# its first social video job. Must match LOCAL_WHISPER_MODEL / LOCAL_WHISPER_CACHE_DIR.
RUN mkdir -p /app/model-cache \
    && python -c "from faster_whisper.utils import download_model; download_model('small', cache_dir='/app/model-cache')"
ENV LOCAL_WHISPER_MODEL=small \
    LOCAL_WHISPER_CACHE_DIR=/app/model-cache
COPY app ./app
EXPOSE 8091
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8091"]
