"""
Pydantic models for the classifier server API.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class TranscriptSegment(BaseModel):
    """Incoming transcript segment from realtime_monitor."""
    call_uuid: str
    role: str
    text: str
    timestamp: float


class ClassificationResult(BaseModel):
    """Classification result sent via WebSocket."""
    call_uuid: str
    classification: Optional[Dict[str, Any]] = None
    last_classification_time: float = 0.0
    is_processing: bool = False
    server_time: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vllm_healthy: bool
    model: str
    env: str
    # Local inference (not OpenAI cloud)
    inference_engine: str = "vLLM"
    model_source: str = "Hugging Face Hub"
    agent_summary: str = ""
    # Where the classifier sends completion requests (same as VLLM_BASE_URL)
    vllm_base_url: str = ""
