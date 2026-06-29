FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU-only (smaller image)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-geometric==2.6.1

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/          ./src/
COPY api/          ./api/
COPY dashboard/    ./dashboard/
COPY models/       ./models/
COPY data/processed/ ./data/processed/

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
