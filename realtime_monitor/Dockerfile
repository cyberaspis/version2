FROM python:3.10-slim

# Install system dependencies required for webrtcvad and other libs
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python dependencies
# We list them here directly for simplicity, or we could copy a requirements.txt
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    uvloop \
    websockets \
    tritonclient[grpc] \
    webrtcvad \
    jinja2 \
    numpy \
    protobuf \
    httpx \
    python-multipart

# Copy application code
COPY ./app /app

# Expose ports
EXPOSE 80 9092

# Command to run the application
# We use uvicorn for the web server, but main.py will likely handle the AudioSocket loop separately
# IF main.py runs the uvicorn server programmatically.
CMD ["python", "main.py"]
