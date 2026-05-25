
import asyncio
import os
import struct
import logging
import uuid
import json
import time
from typing import Dict, List, Optional
import numpy as np
import webrtcvad
import tritonclient.grpc.aio as grpcclient
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn
import httpx
from datetime import datetime

# --- Configuration ---
class Config:
    # General
    LOG_LEVEL = logging.INFO
    TRITON_URL = os.getenv("TRITON_URL", "triton:8001")
    MODEL_NAME = "faster-whisper"
    AUDIOSOCKET_PORT = int(os.getenv("AUDIOSOCKET_PORT", 9092))
    WEB_PORT = int(os.getenv("APP_PORT", 80))
    RECORDINGS_DIR = "/recordings"
    CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8001/push_segment")
    ARI_PROXY_URL  = os.getenv("ARI_PROXY_URL", "http://localhost:8000")

    # Audio
    SAMPLE_RATE = 8000
    FRAME_DURATION_MS = 30
    FRAME_SIZE_BYTES = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0) * 2)
    
    # VAD
    VAD_MODE = 1 # 0-3 (aggressive level)
    VAD_SILENCE_THRESHOLD_FRAMES = int(400 / FRAME_DURATION_MS) # 400ms
    VAD_MIN_SPEECH_DURATION_BYTES = int(SAMPLE_RATE * 2 * 0.2) # 200ms
    VAD_FLUSH_MIN_DURATION_BYTES = int(SAMPLE_RATE * 2 * 0.1) # 100ms
    
    # Normalization
    NORM_ENABLED = True
    NORM_TARGET_RMS = 0.1 # -20dBFS
    NORM_MAX_GAIN = 6.0   # ~15dB
    NORM_LOWER_THRESHOLD = 0.001 # Silence floor

    # Filters
    HALLUCINATIONS = [
        "AUTHORWAVE", "Υπότιτλοι", "Ευχαριστώ για την παρακολούθηση", 
        "Παρακαλώ", "Subtitles"
    ]
    # Short hallucinations text, length threshold in seconds
    SHORT_HALLUCINATIONS = {
        "Ευχαριστώ.": 2.0,
        "Ευχαριστώ": 2.0
    }

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RealTimeApp")

from collections import deque

class GlobalLatencyTracker:
    def __init__(self, window_seconds=60):
        self.history = deque() # (timestamp, latency)
        self.window = window_seconds

    def add(self, latency):
        now = time.time()
        self.history.append((now, latency))
        self._clean(now)

    def _clean(self, now):
        while self.history and self.history[0][0] < now - self.window:
            self.history.popleft()

    def get_average(self):
        self._clean(time.time())
        if not self.history:
            return 0.0
        total = sum(l for _, l in self.history)
        return total / len(self.history)

    def get_median(self):
        self._clean(time.time())
        if not self.history:
            return 0.0
        latencies = sorted([l for _, l in self.history])
        n = len(latencies)
        if n % 2 == 1:
            return latencies[n // 2]
        else:
            return (latencies[n // 2 - 1] + latencies[n // 2]) / 2.0

LATENCY_TRACKER = GlobalLatencyTracker()

# --- Global State ---
class Conversation:
    def __init__(self, uuid_str: str, callee_id: Optional[str] = None):
        self.uuid = uuid_str
        self.callee_id = callee_id  # called number, populated from ARI Proxy
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.active_participants = 0
        self.transcripts = [] # List[dict]

CONVERSATIONS: Dict[str, Conversation] = {}
WEBSOCKETS: List[WebSocket] = []
BACKGROUND_TASKS = set()
TRITON_CLIENT: Optional[grpcclient.InferenceServerClient] = None
CLASSIFIER_CLIENT: Optional[httpx.AsyncClient] = None
CALL_LIST_UPDATE_REQUIRED = False

# --- AudioSocket Protocol ---
TYPE_UUID = 0x01
TYPE_SILENCE = 0x01 
TYPE_AUDIO = 0x10 
TYPE_HANGUP = 0x00

# Constant silent frame (10ms of silence)
SILENT_FRAME = struct.pack(">B H", 0x10, 2 * int(Config.SAMPLE_RATE * 0.01)) + b'\x00' * (2 * int(Config.SAMPLE_RATE * 0.01))

class VADWrapper:
    def __init__(self):
        self.vad = webrtcvad.Vad(Config.VAD_MODE)
        self.buffer = b""
        self.active_buffer = b""
        self.is_speaking = False
        self.silence_counter = 0
        self.total_samples = 0
        self.sample_offset_start = 0

    def process(self, chunk: bytes):
        self.buffer += chunk
        results = []
        while len(self.buffer) >= Config.FRAME_SIZE_BYTES:
            frame = self.buffer[:Config.FRAME_SIZE_BYTES]
            self.buffer = self.buffer[Config.FRAME_SIZE_BYTES:]
            is_speech = self.vad.is_speech(frame, Config.SAMPLE_RATE)
            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.sample_offset_start = self.total_samples
                self.active_buffer += frame
                self.silence_counter = 0
            else:
                if self.is_speaking:
                    self.active_buffer += frame
                    self.silence_counter += 1
                    if self.silence_counter >= Config.VAD_SILENCE_THRESHOLD_FRAMES:
                        if len(self.active_buffer) > Config.VAD_MIN_SPEECH_DURATION_BYTES:
                             ts = self.sample_offset_start / Config.SAMPLE_RATE
                             results.append((self.active_buffer, ts))
                        self.active_buffer = b""
                        self.is_speaking = False
                        self.silence_counter = 0
            self.total_samples += (Config.FRAME_SIZE_BYTES // 2)
        return results

    def flush(self):
        """Flush remaining buffer as a segment if it has minimum length."""
        results = []
        if self.is_speaking and len(self.active_buffer) > Config.VAD_FLUSH_MIN_DURATION_BYTES:
            ts = self.sample_offset_start / Config.SAMPLE_RATE
            results.append((self.active_buffer, ts))
        self.active_buffer = b""
        self.is_speaking = False
        return results

# --- Triton Client ---
def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """Preprocess audio on CPU before sending to Triton (frees GPU instance time).
    Pipeline: INT16 PCM 8kHz → float32 → upsample to 16kHz → RMS normalize.
    """
    # 1. Convert INT16 PCM to float32
    audio_fp32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # 2. Upsample from 8kHz to 16kHz (Whisper's expected sample rate)
    n_orig = len(audio_fp32)
    audio_16k = np.interp(
        np.linspace(0, n_orig - 1, n_orig * 2),
        np.arange(n_orig),
        audio_fp32
    ).astype(np.float32)

    # 3. RMS Normalization
    if Config.NORM_ENABLED:
        rms = np.sqrt(np.mean(audio_16k ** 2))
        if rms > Config.NORM_LOWER_THRESHOLD:
            gain = min(Config.NORM_TARGET_RMS / rms, Config.NORM_MAX_GAIN)
            audio_16k = np.clip(audio_16k * gain, -1.0, 1.0)

    return audio_16k

async def transcribe_audio(audio_bytes: bytes) -> str:
    global TRITON_CLIENT
    try:
        # Preprocess on CPU: INT16 8kHz → float32 16kHz normalized
        audio_16k = preprocess_audio(audio_bytes)
        input_data = audio_16k.reshape(1, -1)

        if TRITON_CLIENT is None:
            logger.info(f"Creating new Triton client for {Config.TRITON_URL}")
            TRITON_CLIENT = grpcclient.InferenceServerClient(url=Config.TRITON_URL)

        inputs = [grpcclient.InferInput("AUDIO", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput("TRANSCRIPT")]

        result = await TRITON_CLIENT.infer(model_name=Config.MODEL_NAME, inputs=inputs, outputs=outputs)
        # Handle batch output [1, 1]
        transcript_batch = result.as_numpy("TRANSCRIPT")
        if transcript_batch is not None and len(transcript_batch) > 0:
            transcript = transcript_batch[0][0] if transcript_batch.ndim > 1 else transcript_batch[0]
            if isinstance(transcript, bytes):
                return transcript.decode('utf-8')
            return str(transcript)
        return ""
    except Exception as e:
        logger.error(f"Triton Error: {e}")
        TRITON_CLIENT = None  # Reset on error
        return ""

# --- WebSocket Manager ---
async def broadcast(message: dict):
    if not WEBSOCKETS:
        return
    msg_type = message.get("type", "unknown")
    if msg_type == "transcript":
        logger.info(f"Broadcasting transcript to {len(WEBSOCKETS)} clients: {message.get('text')}")
    elif msg_type == "call_update":
        logger.info(f"Broadcasting call_update to {len(WEBSOCKETS)} clients (count={len(message.get('calls', []))})")
    else:
        logger.info(f"Broadcasting {msg_type} to {len(WEBSOCKETS)} clients")

    async def send_to_ws(ws):
        try:
            await ws.send_json(message)
            return None
        except Exception as e:
            logger.debug(f"Failed to send to ws: {e}")
            return ws

    results = await asyncio.gather(*(send_to_ws(ws) for ws in WEBSOCKETS))
    to_remove = [ws for ws in results if ws is not None]
    
    for ws in to_remove:
        if ws in WEBSOCKETS:
            WEBSOCKETS.remove(ws)

async def broadcast_call_list():
    global CALL_LIST_UPDATE_REQUIRED
    CALL_LIST_UPDATE_REQUIRED = True

async def throttled_call_list_broadcaster():
    global CALL_LIST_UPDATE_REQUIRED
    logger.info("Throttled broadcaster task started")
    while True:
        try:
            await asyncio.sleep(1.0) # Update at most once per second
            
            # Always broadcast system stats
            system_avg = LATENCY_TRACKER.get_average()
            system_median = LATENCY_TRACKER.get_median()
            
            # Count truly active calls
            active_calls_count = sum(1 for c in CONVERSATIONS.values() if c.active_participants > 0)

            await broadcast({
                "type": "system_stats",
                "system_avg_latency_1min": system_avg,
                "system_median_latency_1min": system_median,
                "active_calls_count": active_calls_count
            })

            now = time.time()

            # Cleanup old conversations (ended more than 5 minutes ago)
            stale = [uuid for uuid, c in CONVERSATIONS.items()
                     if c.end_time and (now - c.end_time > 300)]
            for uuid in stale:
                del CONVERSATIONS[uuid]
                logger.debug(f"Cleaned up stale conversation: {uuid}")

            if CALL_LIST_UPDATE_REQUIRED:
                CALL_LIST_UPDATE_REQUIRED = False
                calls = []
                for c in list(CONVERSATIONS.values()):
                    # Include active calls OR calls ended within the last 60 seconds
                    is_active = c.active_participants > 0
                    is_recently_ended = c.end_time and (now - c.end_time < 60)
                    
                    if is_active or is_recently_ended:
                        calls.append({
                            "uuid": c.uuid,
                            "participants": c.active_participants,
                            "start_time": c.start_time,
                            "end_time": c.end_time,
                            "is_active": is_active,
                            "transcripts": c.transcripts[-100:]
                        })
                
                await broadcast({
                    "type": "call_update", 
                    "calls": calls,
                    "active_calls_count": active_calls_count
                })
        except Exception as e:
            logger.error(f"Broadcaster Task Error: {e}")

# --- Network Handlers ---
async def handle_audiosocket_client(reader, writer):
    cid = str(uuid.uuid4())
    logger.info(f"New connection: {cid}")
    vad = VADWrapper()
    conv: Optional[Conversation] = None
    role = "unknown"

    try:
        while True:
            header = await reader.readexactly(3)
            msg_type = header[0]
            length = struct.unpack(">H", header[1:3])[0]
            payload = await reader.readexactly(length) if length > 0 else b""

            if msg_type == TYPE_UUID:
                call_uuid = ""
                role = "unknown"
                raw_id = "unknown"
                try:
                    logger.info(f"Payload received: len={len(payload)}, content={payload!r}")
                    if len(payload) == 16: # Binary UUID
                        # Use the 16th byte as role indicator
                        role_indicator_byte = payload[15]
                        # Group by the 16 bytes but mask the last one to sustain dashed UUID format
                        masked_payload = bytearray(payload)
                        masked_payload[15] = 0x00
                        call_uuid = str(uuid.UUID(bytes=bytes(masked_payload)))
                        raw_id = payload.hex()
                        
                        if role_indicator_byte == 0: role = "agent"
                        elif role_indicator_byte == 1: role = "caller"
                    else:
                        raw_id = payload.decode('utf-8').strip()
                        if len(raw_id) > 0:
                            role_indicator = raw_id[-1]
                            if role_indicator == '0': 
                                role = "agent"
                                call_uuid = raw_id[:-1]
                            elif role_indicator == '1': 
                                role = "caller"
                                call_uuid = raw_id[:-1]
                            else:
                                call_uuid = raw_id
                        else:
                            call_uuid = "empty"
                    
                    logger.info(f"Parsed ID: {call_uuid} | Role: {role} | Raw: {raw_id}")
                except Exception as e:
                    call_uuid = payload.hex()
                    logger.error(f"UUID Parse Error: {e}, payload_hex={call_uuid}")

                if call_uuid not in CONVERSATIONS:
                    callee_id = await fetch_callee_from_ari(call_uuid)
                    # Re-check after await: a concurrent connection may have created it already.
                    if call_uuid not in CONVERSATIONS:
                        CONVERSATIONS[call_uuid] = Conversation(call_uuid, callee_id=callee_id)
                        logger.info(f"New conversation {call_uuid} | callee={callee_id}")

                conv = CONVERSATIONS[call_uuid]
                if role == "unknown":
                    role = "caller" if conv.active_participants == 0 else "agent"
                
                conv.active_participants += 1
                await broadcast_call_list()
                break
            elif msg_type == TYPE_HANGUP:
                return

        # Keepalive task to prevent Asterisk timeout
        async def keepalive():
            try:
                while True:
                    await asyncio.sleep(1.5)
                    writer.write(SILENT_FRAME)
                    await writer.drain()
            except:
                pass

        keepalive_task = asyncio.create_task(keepalive())

        while True:
            try:
                header = await reader.readexactly(3)
            except asyncio.IncompleteReadError:
                break
            msg_type = header[0]
            length = struct.unpack(">H", header[1:3])[0]
            payload = await reader.readexactly(length) if length > 0 else b""

            if msg_type == TYPE_HANGUP:
                logger.debug(f"Received HANGUP for {cid}")
                break
            elif msg_type >= 0x10 and msg_type <= 0x18:
                segments = vad.process(payload)
                for audio_chunk, ts_offset in segments:
                    logger.info(f"VAD segment detected ({len(audio_chunk)} bytes) for {call_uuid} [{role}]")
                    task = asyncio.create_task(process_transcription(conv, role, audio_chunk, ts_offset))
                    BACKGROUND_TASKS.add(task)
                    task.add_done_callback(BACKGROUND_TASKS.discard)
            else:
                logger.debug(f"Received unknown packet type {msg_type:02x} for {cid}")

    except Exception as e:
        logger.error(f"Connection Error {cid}: {e}")
    finally:
        logger.info(f"Connection closed {cid}")
        # Flush remaining VAD buffer
        final_segments = vad.flush()
        for audio_chunk, ts_offset in final_segments:
            task = asyncio.create_task(process_transcription(conv, role, audio_chunk, ts_offset))
            BACKGROUND_TASKS.add(task)
            task.add_done_callback(BACKGROUND_TASKS.discard)

        if 'keepalive_task' in locals():
            keepalive_task.cancel()
        if conv:
            conv.active_participants -= 1
            if conv.active_participants <= 0:
                conv.end_time = time.time()
                logger.info(f"Call ended: {conv.uuid}")
            await broadcast_call_list()
        writer.close()
        await writer.wait_closed()

os.makedirs(Config.RECORDINGS_DIR, exist_ok=True)

def save_transcript_json(conv: Conversation, role: str):
    def _save():
        filename = os.path.join(Config.RECORDINGS_DIR, f"transcript-{conv.uuid}-{role}.json")
        cid = "inbound" if role == "caller" else "outbound"
        data = {
            "call_id": conv.uuid,
            "callerid": cid, 
            "start_time": datetime.fromtimestamp(conv.start_time).isoformat(),
            "transcripts": [t for t in conv.transcripts if t.get("role") == role]
        }
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
    # Run in background thread
    asyncio.get_event_loop().run_in_executor(None, _save)

async def fetch_callee_from_ari(call_uuid: str) -> Optional[str]:
    """Ask the ARI Proxy for the callee_id of a call by UUID.
    Retries up to 3 times with 500ms delay to handle race condition
    between AudioSocket connection and ARI call registration.
    """
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{Config.ARI_PROXY_URL}/api/calls")
                if resp.status_code == 200:
                    for call in resp.json():
                        if call.get("uuid", "").startswith(call_uuid[:34]):
                            callee = call.get("callee_id")
                            if callee:
                                return callee
        except Exception as e:
            logger.warning(f"ARI Proxy lookup failed for {call_uuid} (attempt {attempt+1}): {e}")
        await asyncio.sleep(0.5)
    return None


async def send_to_classifier(call_uuid: str, role: str, text: str, timestamp: float, callee_id: Optional[str] = None):
    global CLASSIFIER_CLIENT
    try:
        if CLASSIFIER_CLIENT is None:
            CLASSIFIER_CLIENT = httpx.AsyncClient(timeout=5.0)
        await CLASSIFIER_CLIENT.post(Config.CLASSIFIER_URL, json={
            "call_uuid": call_uuid,
            "role": role,
            "text": text,
            "timestamp": timestamp,
            "callee_id": callee_id,
        })
    except Exception as e:
        CLASSIFIER_CLIENT = None  # Reset on error, consistent with TRITON_CLIENT pattern
        logger.error(f"Failed to send to classifier: {e}")

async def process_transcription(conv: Conversation, role: str, audio: bytes, ts_offset: float):
    start_time = time.time()
    text = await transcribe_audio(audio)
    latency = time.time() - start_time
    text = text.strip()
    
    # Filter hallucinations
    if any(h in text for h in Config.HALLUCINATIONS):
        logger.info(f"Filtered hallucination [{role}]: '{text}' for {conv.uuid}")
        return
    
    # Check short hallucinations
    audio_duration_sec = len(audio) / (Config.SAMPLE_RATE * 2) 
    if text in Config.SHORT_HALLUCINATIONS:
        threshold = Config.SHORT_HALLUCINATIONS[text]
        if audio_duration_sec < threshold:
            logger.info(f"Filtered short '{text}' [{role}] (duration {audio_duration_sec:.2f}s < {threshold}s)")
            return
    
    if len(text) < 2:
        logger.debug(f"Filtered too short text: '{text}'")
        return

    LATENCY_TRACKER.add(latency)
    system_avg = LATENCY_TRACKER.get_average()

    data = {
        "type": "transcript",
        "call_uuid": conv.uuid,
        "role": role,
        "text": text,
        "timestamp_secs": ts_offset,
        "confidence": 1.0,
        "latency": latency,
        "audio_duration": audio_duration_sec,
        "system_avg_latency_1min": system_avg
    }
    conv.transcripts.append(data)
    logger.info(f"Transcript [{conv.uuid}][{role}]: {text} (latency: {latency:.2f}s, dur: {audio_duration_sec:.2f}s) [SysAvg: {system_avg:.2f}s]")
    save_transcript_json(conv, role)
    
    # Send to classifier (async background task)
    task = asyncio.create_task(send_to_classifier(conv.uuid, role, text, ts_offset, callee_id=conv.callee_id))
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)

    await broadcast(data)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def get_version():
    return {
        "service": "realtime_monitor",
        "triton_url": Config.TRITON_URL,
        "model_name": Config.MODEL_NAME,
        "vad_mode": Config.VAD_MODE,
        "vad_silence_threshold_frames": Config.VAD_SILENCE_THRESHOLD_FRAMES,
        "norm_enabled": Config.NORM_ENABLED,
        "norm_target_rms": Config.NORM_TARGET_RMS,
        "norm_max_gain": Config.NORM_MAX_GAIN,
    }


@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    WEBSOCKETS.append(websocket)
    try:
        await broadcast_call_list()
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in WEBSOCKETS: WEBSOCKETS.remove(websocket)

@app.on_event("startup")
async def startup_event():
    # Increase thread pool for high-concurrency file I/O
    loop = asyncio.get_event_loop()
    from concurrent.futures import ThreadPoolExecutor
    loop.set_default_executor(ThreadPoolExecutor(max_workers=100))
    
    # Start the throttled broadcaster
    asyncio.create_task(throttled_call_list_broadcaster())

    server = await asyncio.start_server(handle_audiosocket_client, '0.0.0.0', Config.AUDIOSOCKET_PORT)
    logger.info(f"AudioSocket Server listening on {Config.AUDIOSOCKET_PORT}")
    asyncio.create_task(server.serve_forever())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config.WEB_PORT)
