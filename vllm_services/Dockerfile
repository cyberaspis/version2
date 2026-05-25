FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source into a subdirectory to maintain package structure
COPY . vllm_services/

CMD ["python", "-m", "vllm_services.server.app"]
