import httpx
import time
import uuid
import asyncio
import os

# Dashboard (Classifier) base URL — set CLASSIFIER_URL to match your deployment
CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://127.0.0.1:8003")
PUSH_URL = f"{CLASSIFIER_URL.rstrip('/')}/push_segment"

async def simulate_call():
    call_uuid = str(uuid.uuid4())
    url = PUSH_URL
    
    segments = [
        {"role": "agent", "text": "Γεια σας, τηλεφωνώ από την Τράπεζα Πειραιώς."},
        {"role": "caller", "text": "Ναι, παρακαλώ, τι συμβαίνει;"},
        {"role": "agent", "text": "Υπάρχει ένα πρόβλημα με τον λογαριασμό σας. Πρέπει να επιβεβαιώσουμε τα στοιχεία σας."},
        {"role": "agent", "text": "Θα μπορούσατε να μου δώσετε τον κωδικό PIN που μόλις λάβατε στο κινητό σας;"},
        {"role": "caller", "text": "Γιατί το χρειάζεστε αυτό;"},
        {"role": "agent", "text": "Είναι απαραίτητο για την ασφάλεια της συναλλαγής σας. Παρακαλώ δώστε μου το PIN άμεσα."},
    ]
    
    print(f"Push URL: {url}")
    print(f"Starting simulated call: {call_uuid}")
    print("Open the dashboard in your browser to see it live.")
    
    async with httpx.AsyncClient() as client:
        for seg in segments:
            payload = {
                "call_uuid": call_uuid,
                "role": seg["role"],
                "text": seg["text"],
                "timestamp": time.time()
            }
            try:
                response = await client.post(url, json=payload)
                print(f"Pushed segment: {seg['text'][:30]}... | Status: {response.status_code}")
            except Exception as e:
                print(f"Error pushing segment: {e}")
            
            await asyncio.sleep(2)  # Wait 2 seconds between segments

if __name__ == "__main__":
    asyncio.run(simulate_call())
