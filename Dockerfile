# Streamlit app image for Railway. The runtime needs only the app code, src/, and
# the processed JSONL artifacts + a populated Pinecone index — no raw PDFs, no
# embedding model (Pinecone does inference), so the image stays light.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production

WORKDIR /app

# Dependencies first (cached layer). requirements.txt is exported from uv.lock.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application + runtime data (data/processed/extracted and _enrich are excluded via .dockerignore).
COPY app.py README.md ./
COPY src ./src
COPY data/processed ./data/processed
COPY data/golden ./data/golden

EXPOSE 8501

# Exec form + sh -c so ${PORT} (injected by Railway) is substituted; defaults to 8501 locally.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
