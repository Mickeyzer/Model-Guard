FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (Note: the data directory will be mounted via docker-compose)
COPY . ./middleware

EXPOSE 8000

CMD ["uvicorn", "middleware.app:app", "--host", "0.0.0.0", "--port", "8000"]
