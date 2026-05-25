
import asyncio
import websockets
import json
import sys
import time

async def check_parity(url, duration=90):
    print(f"Connecting to {url}...")
    transcripts_received = []
    calls_seen = set()
    
    try:
        async with websockets.connect(url) as websocket:
            print("Connected. Listening...")
            start_time = time.time()
            
            while time.time() - start_time < duration:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    mtype = data.get("type")
                    
                    if mtype == "transcript":
                        text = data.get("text")
                        role = data.get("role")
                        call_id = data.get("call_uuid")
                        print(f"[{time.strftime('%H:%M:%S')}] [TRANSCRIPT] {call_id[:8]} [{role}]: {text}")
                        transcripts_received.append(data)
                    elif mtype == "call_update":
                        for c in data.get("calls", []):
                            if c["uuid"] not in calls_seen:
                                print(f"[{time.strftime('%H:%M:%S')}] [CALL_NEW] {c['uuid'][:8]} | P: {c['participants']}")
                                calls_seen.add(c["uuid"])
                            # Check if history sync is working
                            if len(c.get("transcripts", [])) > 0:
                                print(f"[{time.strftime('%H:%M:%S')}] [HISTORY] {c['uuid'][:8]} has {len(c['transcripts'])} items")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error: {e}")
                    break
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print(f"\nCollected {len(transcripts_received)} transcripts from {len(calls_seen)} calls.")
    with open("/tmp/ui_sim_full.json", "w") as f:
        json.dump({"transcripts": transcripts_received, "calls": list(calls_seen)}, f, indent=2)

if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8080/ws"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    asyncio.run(check_parity(target_url, duration))
