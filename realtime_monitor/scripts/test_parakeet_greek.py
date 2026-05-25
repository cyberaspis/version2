#!/usr/bin/env python3
"""
Quick test script to evaluate NeMo Parakeet-TDT 0.6B on Greek audio.
This will help determine if the model quality is acceptable before full migration.
"""
import time
import sys

def test_parakeet():
    print("=" * 60)
    print("NeMo Parakeet-TDT 0.6B - Greek Language Test")
    print("=" * 60)
    
    # Import NeMo (will download model on first run)
    print("\n[1/4] Loading NeMo ASR module...")
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("ERROR: NeMo not installed. Install with:")
        print("  pip install 'nemo_toolkit[asr]'")
        sys.exit(1)
    
    # Load model
    print("\n[2/4] Loading Parakeet-TDT 0.6B model (~2.5GB download on first run)...")
    start = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    print(f"       Model loaded in {time.time() - start:.1f}s")
    
    # Check supported languages
    print("\n[3/4] Model Info:")
    print(f"       Type: {type(model).__name__}")
    if hasattr(model, 'cfg'):
        print(f"       Sample Rate: {model.cfg.get('sample_rate', 'N/A')}")
    
    # Test with a sample audio file if provided
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"\n[4/4] Transcribing: {audio_path}")
        start = time.time()
        
        # Transcribe
        transcriptions = model.transcribe([audio_path])
        elapsed = time.time() - start
        
        print(f"\n{'=' * 60}")
        print("RESULT:")
        print(f"{'=' * 60}")
        for i, text in enumerate(transcriptions):
            print(f"  [{i}] {text}")
        print(f"\n  Latency: {elapsed:.2f}s")
        print(f"{'=' * 60}")
    else:
        print("\n[4/4] No audio file provided. Usage:")
        print("       python test_parakeet_greek.py /path/to/greek_audio.wav")
    
    print("\nDone!")

if __name__ == "__main__":
    test_parakeet()
