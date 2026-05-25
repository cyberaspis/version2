import asyncio
import time
from typing import List, Dict, Any
from classifier import VishingClassifier
from vllm_client import VLLMClient
import config

# We'll use a mocked version of CallBuffer logic to simulate production behavior
class SimulatedCallBuffer:
    def __init__(self, call_uuid: str):
        self.call_uuid = call_uuid
        self.segments: List[dict] = []
        self.last_update = time.time()
        self.last_classification_time = 0.0
        self.is_processing = False

    def add_segment(self, text: str, role: str):
        self.segments.append({"text": text, "role": role})
        self.last_update = time.time()

    def get_full_text(self) -> str:
        return " ".join([s["text"] for s in self.segments])

    def should_classify(self) -> bool:
        if self.is_processing:
            return False
        
        words = self.get_full_text().split()
        word_count = len(words)
        time_since_last = time.time() - self.last_classification_time
        
        # Mimic production triggers
        if word_count >= config.MIN_WORDS_FOR_CLASSIFICATION and time_since_last > config.DEBOUNCE_SECONDS:
            return True
        return False

async def simulate_call(transcript: str, call_id: str = "sim-001"):
    """
    Simulates a call by breaking the transcript into segments and pushing them 
    incrementally, just like Whisper would do in production.
    """
    print(f"\n>>> Starting Streaming Simulation for Call: {call_id}")
    
    # Initialize real production components
    client = VLLMClient(base_url=config.VLLM_BASE_URL, model_name=config.MODEL_NAME)
    classifier = VishingClassifier(client)
    buffer = SimulatedCallBuffer(call_id)
    
    # Split transcript into sentences/segments
    segments = transcript.split(". ")
    
    for i, seg_text in enumerate(segments):
        role = "caller" if i % 2 == 0 else "agent"
        print(f"\n[Segment {i+1}] {role.upper()}: {seg_text[:60]}...")
        
        # Add segment to buffer
        buffer.add_segment(seg_text, role)
        
        # Check if classification should trigger (just like production server/app.py)
        if buffer.should_classify():
            print("--- Triggering LLM Analysis ---")
            buffer.is_processing = True
            
            # Run classification on current cumulative text
            result = await classifier.classify(buffer.get_full_text(), call_id=call_id)
            
            buffer.last_classification_time = time.time()
            buffer.is_processing = False
            
            # Display results
            score = result.get("risk_score", 0)
            status = result.get("risk_status", "SAFE")
            prob = result.get("prob_yes", 0)
            print(f"RESULT: Status={status} | Points={score:.1f} | LLM Prob={prob:.2f}")
        else:
            print("(Waiting for more data or debounce...)")
        
        # Simulate small delay between segments
        await asyncio.sleep(0.5)

    print("\n>>> Simulation Completed.")

if __name__ == "__main__":
    # Example "Vishing" transcript for testing
    vishing_example = (
        "Hello, this is Bank of Greece security department. "
        "We have detected an unauthorized transaction of 1200 Euros on your account. "
        "To block this transfer, we need to verify your identity immediately. "
        "Please provide the 6-digit verification code you just received on your mobile phone. "
        "We also need your e-banking username to access and secure the account. "
        "If you don't do this now, the funds will be lost forever. "
        "Thank you for your cooperation."
    )
    
    asyncio.run(simulate_call(vishing_example))
