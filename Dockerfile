FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOUBAO_WORK_HOST=0.0.0.0 \
    DOUBAO_WORK_PORT=18789

WORKDIR /app
COPY pyproject.toml README.md ./
COPY doubao_work2api ./doubao_work2api
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/accounts /app/data \
    && chown -R app:app /app
USER app

EXPOSE 18789
CMD ["python", "-m", "doubao_work2api"]
