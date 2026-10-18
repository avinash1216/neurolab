# Python version matches the environment the tests and end-to-end checks ran on.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Trained artifacts and demo trials are baked in; no dataset download at runtime.
COPY .streamlit ./.streamlit
COPY config ./config
COPY src ./src
COPY app ./app
COPY artifacts ./artifacts
COPY reports ./reports

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.fileWatcherType=none"]
