import os
import json
import uuid as _uuid
import httpx
import asyncio
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Set, Dict

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARI_URL = os.getenv("ARI_URL", "http://127.0.0.1:8088/ari")
ARI_USER = os.getenv("ARI_USER", "whisper")
ARI_PASS = os.getenv("ARI_PASS", "")
ARI_APP  = os.getenv("ARI_APP",  "simulation-app")
META_LOG_FILE = os.getenv("META_LOG_FILE", "/var/log/asterisk/audiosocket_meta.json")
AUDIO_DIR = os.getenv("AUDIO_DIR", "/data/audio")

class CallInfo(BaseModel):
    channel_id: str
    all_channels: List[str] = []
    caller_id: str
    callee_id: str
    uuid: Optional[str]
    type: str
    role: str
    timestamp: int

# --- Performance Optimized State Management ---
GLOBAL_CALLS: Dict[str, CallInfo] = {}  # UUID -> CallInfo
CHANNEL_TO_UUID: Dict[str, str] = {}    # Channel Name -> UUID
LOCK = asyncio.Lock()

async def tail_metadata_log():
    """Background task to follow the metadata log file and update state."""
    print(f"Starting metadata log tailer on {META_LOG_FILE}")
    last_info = os.stat(META_LOG_FILE) if os.path.exists(META_LOG_FILE) else None
    
    # Start by reading existing file to populate state
    if os.path.exists(META_LOG_FILE):
        with open(META_LOG_FILE, 'r') as f:
            for line in f:
                try:
                    process_log_entry(json.loads(line))
                except:
                    continue

    file_handle = open(META_LOG_FILE, 'r')
    file_handle.seek(0, os.SEEK_END)

    while True:
        line = file_handle.readline()
        if not line:
            await asyncio.sleep(0.5)
            continue
        try:
            process_log_entry(json.loads(line))
        except:
            continue

def process_log_entry(entry: dict):
    """Update internal state from a log entry."""
    channel_name = entry.get('channel')
    uuid = entry.get('base_uuid', entry.get('uuid'))
    if not uuid or not channel_name:
        return

    role = entry.get('role', 'unknown')
    
    if uuid not in GLOBAL_CALLS:
        GLOBAL_CALLS[uuid] = CallInfo(
            channel_id="", # To be filled by ARI mapping
            all_channels=[],
            caller_id=entry.get('caller_id', 'unknown'),
            callee_id=entry.get('callee_id', 'unknown'),
            uuid=uuid,
            type=entry.get('type', 'live'),
            role=role,
            timestamp=entry.get('timestamp', 0)
        )
    
    CHANNEL_TO_UUID[channel_name] = uuid

async def sync_with_ari():
    """Poll ARI periodically to prune dead calls and map IDs."""
    async with httpx.AsyncClient(auth=(ARI_USER, ARI_PASS)) as client:
        while True:
            try:
                resp = await client.get(f"{ARI_URL}/channels", timeout=5)
                if resp.status_code == 200:
                    ari_channels = resp.json()
                    active_ids = {c['id'] for c in ari_channels}
                    name_to_id = {c['name']: c['id'] for c in ari_channels}
                    
                    async with LOCK:
                        # 1. Update mappings and track active UUIDs
                        active_uuids_this_cycle = set()
                        for name, chan_id in name_to_id.items():
                            uuid = CHANNEL_TO_UUID.get(name)
                            if not uuid and ';' in name:
                                base, leg = name.split(';')
                                other_leg = "1" if leg == "2" else "2"
                                uuid = CHANNEL_TO_UUID.get(f"{base};{other_leg}")
                            
                            if uuid and uuid in GLOBAL_CALLS:
                                active_uuids_this_cycle.add(uuid)
                                if chan_id not in GLOBAL_CALLS[uuid].all_channels:
                                    GLOBAL_CALLS[uuid].all_channels.append(chan_id)
                                
                                # Selection Logic:
                                # We want channel_id to point to the leg that matches the metadata name
                                # Or prioritize the leg that was actually tagged in the log
                                if name in CHANNEL_TO_UUID:
                                    GLOBAL_CALLS[uuid].channel_id = chan_id

                        # 2. Prune dead calls
                        dead_uuids = [u for u in GLOBAL_CALLS if u not in active_uuids_this_cycle]
                        for uuid, call in GLOBAL_CALLS.items():
                            if uuid not in dead_uuids:
                                # Clean up stale channel IDs from the list
                                call.all_channels = [cid for cid in call.all_channels if cid in active_ids]
                                if not call.channel_id or call.channel_id not in active_ids:
                                    if call.all_channels:
                                        call.channel_id = call.all_channels[0]
                        
                        for uuid in dead_uuids:
                            del GLOBAL_CALLS[uuid]
                        
                        for uuid in dead_uuids:
                            del GLOBAL_CALLS[uuid]
                            # Clean CHANNEL_TO_UUID mapping (optional but good for RAM)
                            # CHANNEL_TO_UUID = {k: v for k, v in CHANNEL_TO_UUID.items() if v != uuid}
                
                # Broadcast update if anyone is connected
                await manager.broadcast(get_calls_json())

            except Exception as e:
                print(f"ARI Sync Error: {e}")
            
            await asyncio.sleep(2)

def get_calls_json():
    return json.dumps([c.dict() for c in GLOBAL_CALLS.values()])

# --- WebSocket Management ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        if not self.active_connections: return
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

async def _ari_ws_worker():
    """Maintain WebSocket connection to Asterisk ARI to keep simulation-app registered."""
    base = ARI_URL.replace("http://", "ws://").replace("https://", "wss://")
    base = base[: base.rfind("/ari")] if "/ari" in base else base
    ws_url = f"{base}/ari/events?api_key={ARI_USER}:{ARI_PASS}&app={ARI_APP}&subscribeAll=true"
    while True:
        try:
            async with websockets.connect(ws_url, origin=None) as ws:
                print(f"[ARI WS] Connected — simulation-app registered", flush=True)
                async for message in ws:
                    try:
                        data = json.loads(message)
                        if data.get("type") == "StasisStart":
                            print(f"[ARI WS] StasisStart: {data.get('channel', {}).get('id', '?')}", flush=True)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[ARI WS] Disconnected: {e} — reconnecting in 3s", flush=True)
            await asyncio.sleep(3)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(tail_metadata_log())
    asyncio.create_task(sync_with_ari())
    asyncio.create_task(_ari_ws_worker())

@app.websocket("/ws/calls")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(get_calls_json())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except:
        manager.disconnect(websocket)

# --- REST Endpoints (High Speed) ---
@app.get("/api/calls", response_model=List[CallInfo])
async def get_calls():
    return list(GLOBAL_CALLS.values())

@app.post("/api/play/{channel_id}")
async def play_sound(channel_id: str, sound: str = "hello-world"):
    async with httpx.AsyncClient(auth=(ARI_USER, ARI_PASS)) as client:
        try:
            # 1. Get channel name
            resp = await client.get(f"{ARI_URL}/channels/{channel_id}")
            resp.raise_for_status()
            channel_name = resp.json().get('name', channel_id)

            # 2. Originate
            payload = {
                "endpoint": "Local/s@ari-playback-spy",
                "extension": "s",
                "context": "ari-play-sound",
                "priority": 1,
                "variables": {"TARGET_CHANNEL": channel_name, "PLAY_SOUND": sound}
            }
            resp = await client.post(f"{ARI_URL}/channels", json=payload)
            resp.raise_for_status()
            return {"status": "success", "message": f"Playback started on {channel_id}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/play/uuid/{uuid}")
async def play_sound_by_uuid(uuid: str, sound: str = "hello-world"):
    if uuid not in GLOBAL_CALLS:
        raise HTTPException(status_code=404, detail="UUID not found")
    
    call = GLOBAL_CALLS[uuid]
    if not call.channel_id:
        raise HTTPException(status_code=400, detail="No active channel for this UUID")
        
    return await play_sound(call.channel_id, sound)

@app.post("/api/hangup")
async def hangup_call(channels: List[str]):
    async with httpx.AsyncClient(auth=(ARI_USER, ARI_PASS)) as client:
        tasks = [client.delete(f"{ARI_URL}/channels/{cid}") for cid in channels]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return {"status": "completed", "results": str(responses)}

@app.post("/api/hangup/uuid/{uuid}")
async def hangup_call_by_uuid(uuid: str):
    if uuid not in GLOBAL_CALLS:
        raise HTTPException(status_code=404, detail="UUID not found")
    
    call = GLOBAL_CALLS[uuid]
    return await hangup_call(call.all_channels)

# ---------------------------------------------------------------------------
# Simulation API
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    conversation_id: str
    call_uuid: Optional[str] = None  # generated with "00" suffix if omitted

class SimulateStatus(BaseModel):
    call_uuid: str
    conversation_id: str
    status: str   # starting | playing | done | error
    error: Optional[str] = None

ACTIVE_SIMULATIONS: Dict[str, dict] = {}


async def _poll_playback(pb_id: str, timeout: int = 1800) -> None:
    """Wait until ARI reports a playback as done/failed/gone."""
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(auth=(ARI_USER, ARI_PASS), timeout=5) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(f"{ARI_URL}/playbacks/{pb_id}")
                if resp.status_code == 404:
                    return  # gone = finished
                if resp.is_success and resp.json().get("state") in ("done", "failed"):
                    return
            except Exception:
                pass
            await asyncio.sleep(2)


async def _run_simulation(call_uuid: str, conversation_id: str,
                          caller_id: str, callee_id: str) -> None:
    """Background task: bridge + channels + playback + cleanup."""
    sim = ACTIVE_SIMULATIONS[call_uuid]
    bridge_id = chan_a = chan_b = None

    try:
        async with httpx.AsyncClient(auth=(ARI_USER, ARI_PASS), timeout=10) as client:
            # Bridge
            r = await client.post(f"{ARI_URL}/bridges", params={"type": "mixing"})
            bridge_id = r.json().get("id")
            if not bridge_id:
                raise RuntimeError("Failed to create bridge")
            sim["bridge_id"] = bridge_id

            # UUID masking mirrors run_audiosocket_stress_random.py:
            #   conv_id ends in "00"  → dialplan FINAL_ID = conv_id[0:34]+"00" = conv_id  (ROLE=inbound)
            #   conv_id[:-1]+"1"      → dialplan FINAL_ID = conv_id[0:34]+"01"            (ROLE=outbound)
            # So quality_service registration with ari_conv_id=call_uuid matches inbound leg exactly.
            conv_id_inbound  = call_uuid
            conv_id_outbound = call_uuid[:-1] + "1"

            r_a = await client.post(f"{ARI_URL}/channels",
                params={"endpoint": "Local/s@simulate_context/n",
                        "app": ARI_APP, "appArgs": "caller", "formats": "ulaw"},
                json={"variables": {"__CONV_ID": conv_id_inbound,  "ROLE": "inbound",
                                    "CALLER_ID": caller_id, "CALLEE_ID": callee_id}})
            chan_a = r_a.json().get("id")

            r_b = await client.post(f"{ARI_URL}/channels",
                params={"endpoint": "Local/s@simulate_context/n",
                        "app": ARI_APP, "appArgs": "callee", "formats": "ulaw"},
                json={"variables": {"__CONV_ID": conv_id_outbound, "ROLE": "outbound",
                                    "CALLER_ID": caller_id, "CALLEE_ID": callee_id}})
            chan_b = r_b.json().get("id")

            if not chan_a or not chan_b:
                raise RuntimeError("Channel creation failed")
            sim["chan_a"] = chan_a
            sim["chan_b"] = chan_b

            await asyncio.sleep(0.5)

            await client.post(f"{ARI_URL}/bridges/{bridge_id}/addChannel",
                              params={"channel": f"{chan_a},{chan_b}"})

            # Wait for AudioSocket to establish before starting playback
            await asyncio.sleep(2.0)

            sim["status"] = "playing"

            audio_base = os.path.join(AUDIO_DIR, conversation_id)
            spk1 = os.path.join(audio_base, "speaker_1_filtered")  # scammer (caller)
            spk2 = os.path.join(audio_base, "speaker_2_filtered")  # victim (callee)

            # chan_a ends "00" → CALLEE in realtime_monitor → victim (spk2)
            # chan_b ends "01" → CALLER in realtime_monitor → scammer (spk1)
            r_pb_a = await client.post(f"{ARI_URL}/channels/{chan_a}/play",
                                       params={"media": f"sound:{spk2}"})
            r_pb_b = await client.post(f"{ARI_URL}/channels/{chan_b}/play",
                                       params={"media": f"sound:{spk1}"})

            pb_id_a = r_pb_a.json().get("id") if r_pb_a.is_success else None
            pb_id_b = r_pb_b.json().get("id") if r_pb_b.is_success else None

        # Wait for both playbacks outside the client context
        polls = []
        if pb_id_a:
            polls.append(_poll_playback(pb_id_a))
        if pb_id_b:
            polls.append(_poll_playback(pb_id_b))
        if polls:
            await asyncio.gather(*polls)

        await asyncio.sleep(5)  # grace period for final VAD flushes
        sim["status"] = "done"

    except Exception as e:
        sim["status"] = "error"
        sim["error"] = str(e)
        print(f"[simulate] {call_uuid} error: {e}")

    finally:
        async with httpx.AsyncClient(auth=(ARI_USER, ARI_PASS), timeout=5) as client:
            if bridge_id:
                await client.delete(f"{ARI_URL}/bridges/{bridge_id}")
            if chan_a:
                await client.delete(f"{ARI_URL}/channels/{chan_a}")
            if chan_b:
                await client.delete(f"{ARI_URL}/channels/{chan_b}")


@app.post("/api/simulate", response_model=SimulateStatus, status_code=202)
async def simulate_call(body: SimulateRequest, background_tasks: BackgroundTasks):
    """Trigger a simulated call from /data/audio through the full pipeline."""
    audio_path = os.path.join(AUDIO_DIR, body.conversation_id)
    if not os.path.isdir(audio_path):
        raise HTTPException(status_code=404,
                            detail=f"Conversation '{body.conversation_id}' not found in {AUDIO_DIR}")

    caller_id = callee_id = "unknown"
    meta_path = os.path.join(audio_path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        caller_id = str(meta.get("caller_id", "unknown"))
        callee_id = str(meta.get("callee_id", "unknown"))

    # Force "00" suffix so FINAL_ID of inbound leg == call_uuid (clean quality_service match)
    if body.call_uuid:
        call_uuid = body.call_uuid
    else:
        call_uuid = str(_uuid.uuid4())[:-2] + "00"

    ACTIVE_SIMULATIONS[call_uuid] = {
        "call_uuid": call_uuid,
        "conversation_id": body.conversation_id,
        "status": "starting",
        "error": None,
        "bridge_id": None,
        "chan_a": None,
        "chan_b": None,
    }

    background_tasks.add_task(_run_simulation, call_uuid, body.conversation_id,
                               caller_id, callee_id)

    return SimulateStatus(call_uuid=call_uuid, conversation_id=body.conversation_id,
                          status="starting")


@app.get("/api/simulate/{call_uuid}", response_model=SimulateStatus)
async def get_simulation_status(call_uuid: str):
    """Poll the status of a running or completed simulation."""
    sim = ACTIVE_SIMULATIONS.get(call_uuid)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return SimulateStatus(call_uuid=sim["call_uuid"], conversation_id=sim["conversation_id"],
                          status=sim["status"], error=sim.get("error"))


@app.get("/api/audio/conversations")
async def list_audio_conversations():
    """List available conversations in AUDIO_DIR with their metadata."""
    if not os.path.isdir(AUDIO_DIR):
        return []

    # Build class lookup from _manifest.json if present
    class_map = {}
    manifest_path = os.path.join(AUDIO_DIR, "_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for entry in json.load(f):
                cid = entry.get("conversation_id", "")
                class_map[cid] = entry.get("class", "unknown")
                class_map[f"conversation_{cid}"] = entry.get("class", "unknown")

    convs = []
    for entry in os.scandir(AUDIO_DIR):
        if not entry.is_dir():
            continue
        if not (os.path.exists(os.path.join(entry.path, "speaker_1_filtered.wav")) and
                os.path.exists(os.path.join(entry.path, "speaker_2_filtered.wav"))):
            continue
        meta = {}
        meta_path = os.path.join(entry.path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        convs.append({
            "conversation_id": entry.name,
            "caller_id":      meta.get("caller_id",  "unknown"),
            "callee_id":      meta.get("callee_id",  "unknown"),
            "classification": class_map.get(entry.name, "unknown"),
        })
    return convs


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
