FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY wheels ./wheels
RUN pip install --no-index --find-links=./wheels -r requirements.txt && rm -rf ./wheels

RUN useradd --create-home --uid 1000 --shell /bin/bash raguser

COPY src ./src
COPY tests ./tests
COPY data ./data

RUN mkdir -p /app/data /app/chroma_db \
    && chown -R raguser:raguser /app

USER raguser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
