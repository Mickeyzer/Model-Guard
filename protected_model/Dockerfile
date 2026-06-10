FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Bake the trained model into the image at build time
RUN python train_and_save.py

# Expose port
EXPOSE 8001

# Healthcheck dependency
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
