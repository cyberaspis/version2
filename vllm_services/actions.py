"""
Manager for automated actions (audio playback, hangup) via external REST API.
"""

import logging
import time
from typing import Optional
import httpx
from . import config

logger = logging.getLogger("ActionManager")

class ActionManager:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or config.ACTIONS_API_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=10.0
        )
        # Tracking triggered actions to avoid duplicates/spam
        # call_uuid -> set of triggered levels
        self._triggered: dict[str, set[str]] = {}

    async def close(self):
        await self._client.aclose()

    async def trigger_action(self, call_uuid: str, risk_status: str) -> Optional[dict]:
        """
        Trigger an action based on the risk status.
        
        Returns a dict with action info if an action was triggered, else None.
        """
        if not call_uuid:
            return None

        if call_uuid not in self._triggered:
            self._triggered[call_uuid] = set()

        if risk_status == "VISHING" and "VISHING" not in self._triggered[call_uuid]:
            await self._play_sound(call_uuid, "vishing-detected")
            self._triggered[call_uuid].add("VISHING")
            return {"type": "SOUND", "detail": "vishing-detected", "timestamp": time.time()}

        return None

    async def _play_sound(self, call_uuid: str, sound: str):
        url = f"/api/play/uuid/{call_uuid}"
        params = {"sound": sound}
        try:
            logger.info(f"Triggering sound '{sound}' for call {call_uuid}")
            response = await self._client.post(url, params=params)
            response.raise_for_status()
            logger.info(f"Sound triggered successfully: {response.json()}")
        except Exception as e:
            logger.error(f"Failed to trigger sound: {e}")

    async def _hangup(self, call_uuid: str):
        url = f"/api/hangup/uuid/{call_uuid}"
        try:
            logger.info(f"Triggering hangup for call {call_uuid}")
            response = await self._client.post(url)
            response.raise_for_status()
            logger.info(f"Hangup triggered successfully: {response.json()}")
        except Exception as e:
            logger.error(f"Failed to trigger hangup: {e}")

    def cleanup(self, call_uuid: str):
        if call_uuid in self._triggered:
            del self._triggered[call_uuid]
