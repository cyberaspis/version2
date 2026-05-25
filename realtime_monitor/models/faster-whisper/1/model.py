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
            # self.model = WhisperModel("sam8000/whisper-large-v3-turbo-greek-greece", device="cuda", compute_type="int8_float16")
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
