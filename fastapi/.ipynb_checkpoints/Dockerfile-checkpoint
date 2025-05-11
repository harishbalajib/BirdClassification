FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y ffmpeg

# Install Python deps early (separate COPY for caching)
COPY base-requirements.txt .
RUN pip install --no-cache-dir -r base-requirements.txt

COPY dev-requirements.txt .
RUN pip install --no-cache-dir -r dev-requirements.txt

# Now copy actual app
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
