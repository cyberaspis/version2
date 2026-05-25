FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for some pip packages)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables with defaults
ENV ARI_URL=http://127.0.0.1:8088/ari
ENV ARI_USER=whisper
ENV ARI_PASS=

EXPOSE 8000

CMD ["python", "server.py"]
