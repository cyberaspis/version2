import numpy as np
import requests
import json

# Create a small dummy audio chunk (1 second of silence at 16kHz)
audio = np.zeros((1, 16000), dtype=np.float32)

# Triton HTTP endpoint
url = "http://localhost:8000/v2/models/faster-whisper/infer"

# Payload
payload = {
    "inputs": [
        {
            "name": "AUDIO",
            "shape": [1, 16000],
            "datatype": "FP32",
            "data": audio.tolist()
        }
    ]
}

print("Sending request to Triton...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Success!")
    result = response.json()
    transcript = result['outputs'][0]['data'][0]
    print(f"Transcript: '{transcript}'")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
