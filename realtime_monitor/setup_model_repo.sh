#!/bin/bash
mkdir -p models/faster-whisper/1

# 1. Create config.pbtxt
cat > models/faster-whisper/config.pbtxt <<EOF
name: "faster-whisper"
backend: "python"
max_batch_size: 32
dynamic_batching {
  preferred_batch_size: [ 8, 16, 24, 32 ]
  max_queue_delay_microseconds: 200000
}

input [
  {
    name: "AUDIO"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
output [
  {
    name: "TRANSCRIPT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

instance_group [
  {
    count: 22
    kind: KIND_GPU
  }
]
EOF

# 2. Create model.py
cat > models/faster-whisper/1/model.py <<EOF
import triton_python_backend_utils as pb_utils
import numpy as np
import json
try:
    from faster_whisper import WhisperModel
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper"])
    from faster_whisper import WhisperModel

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        # Load Model (large-v3-turbo for much faster inference)
        try:
            self.model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
        except Exception as e:
            pb_utils.Logger.log_error(f"FATAL ERROR LOADING MODEL: {e}")
            raise e

    def execute(self, requests):
        responses = []
        for request in requests:
            # Input is pre-processed float32 16kHz audio (preprocessing done client-side)
            input_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO")
            audio_16k = input_tensor.as_numpy().flatten().astype(np.float32)

            # Transcribe (pure inference only)
            segments, info = self.model.transcribe(audio_16k, beam_size=1, language="el")
            text = " ".join([s.text for s in segments])

            # Create Output
            output_tensor = pb_utils.Tensor("TRANSCRIPT", np.array([text.encode('utf-8')], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        pass
EOF

echo "Model repository created at ./models"
