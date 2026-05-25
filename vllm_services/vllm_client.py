"""
Async client for local vLLM serving Mistral-7B-Instruct (Hugging Face model).

vLLM exposes HTTP routes with an OpenAI-*like* JSON shape; requests go to your
GPU server (e.g. http://127.0.0.1:8010), not to OpenAI's cloud.

Provides:
- Raw completions (complete)
- Logprobs-based YES/NO classification (classify_yes_no)
- JSON-based classification (classify_json)
- Health check
"""

import json
import logging
import math
import re
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from . import config

logger = logging.getLogger(__name__)


class VLLMClient:
    """
    Async client for local vLLM (Mistral-7B etc.).

    Uses httpx.AsyncClient with connection pooling for efficient
    concurrent requests.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or config.VLLM_BASE_URL).rstrip("/")
        self.model = model or config.MODEL_NAME
        self.timeout = timeout or config.VLLM_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
            )
        return self._client

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the vLLM server is reachable (try /health then GET /v1/models)."""
        client = await self._get_client()
        for path in ("/health", "/v1/models"):
            try:
                response = await client.get(path)
                if response.status_code == 200:
                    return True
            except Exception as e:
                logger.debug(f"Health check {path}: {e}")
                continue
        logger.warning(
            "Health check failed: /health and /v1/models unreachable at %s — is vLLM running?",
            self.base_url,
        )
        return False

    async def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[int] = None,
        stop: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion request to the vLLM server.

        Returns the raw API response as a dict.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens or config.VLLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.VLLM_TEMPERATURE,
        }
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if stop:
            payload["stop"] = stop

        client = await self._get_client()
        try:
            response = await client.post("/v1/completions", json=payload)
        except httpx.RequestError as e:
            raise RuntimeError(
                f"vLLM unreachable at {self.base_url} ({e}). "
                "Start it first: bash run_vllm.sh (default port 8010). "
                "If it runs on another host/port, export VLLM_BASE_URL before run_dashboard.sh."
            ) from e
        if response.status_code >= 400:
            # vLLM returns 400 when prompt exceeds context (max_model_len) among other cases
            try:
                logger.warning(
                    "vLLM completions error %s: %s",
                    response.status_code,
                    response.text[:1000],
                )
            except Exception:
                logger.warning("vLLM completions error %s (body unreadable)", response.status_code)
            try:
                err = response.json()
                msg = err.get("error", {}).get("message") or response.text
            except Exception:
                msg = response.text
            raise httpx.HTTPStatusError(
                f"{response.status_code} {msg[:500]}",
                request=response.request,
                response=response,
            )
        response.raise_for_status()
        return response.json()

    async def complete_via_chat(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Same as complete() but uses /v1/chat/completions (for vLLM that only exposes chat).
        Returns a response shaped like completions so choices[0]["text"] is the reply.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or config.VLLM_MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.VLLM_TEMPERATURE,
            "stream": False,
        }
        client = await self._get_client()
        try:
            response = await client.post("/v1/chat/completions", json=payload)
        except httpx.RequestError as e:
            raise RuntimeError(
                f"vLLM unreachable at {self.base_url} ({e}). "
                "Start it first: bash run_vllm.sh (default port 8010). "
                "Or set VLLM_BASE_URL to your vLLM server URL."
            ) from e
        if response.status_code == 500:
            try:
                err_body = response.json()
                logger.warning(f"vLLM chat 500: {err_body}")
            except Exception:
                logger.warning(f"vLLM chat 500: {response.text[:500]}")
        response.raise_for_status()
        data = response.json()
        # Normalize to completion shape: choices[0].text (handle None content)
        for c in data.get("choices", []):
            msg = c.get("message") or c.get("delta") or {}
            content = msg.get("content")
            c["text"] = content if content is not None else ""
        return data

    async def classify_yes_no(self, prompt: str) -> Tuple[float, float]:
        """
        Classify a prompt using logprobs-based YES/NO extraction.

        Requests a single token completion with logprobs, then extracts
        the log-probabilities for YES and NO tokens. Applies softmax
        to get P(YES).

        Returns:
            (prob_yes, latency_ms)
        """
        start = time.perf_counter()

        result = await self.complete(
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=config.VLLM_LOGPROBS,
        )

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Extract logprobs from the response
        try:
            choice = result["choices"][0]
            top_logprobs_list = choice.get("logprobs", {}).get("top_logprobs", [])

            if not top_logprobs_list:
                # Fallback: parse the generated text
                generated = choice.get("text", "").strip().upper()
                prob_yes = 1.0 if generated.startswith("YES") else 0.0
                return prob_yes, latency_ms

            # top_logprobs_list[0] is a dict of {token: logprob} for the first generated token
            token_logprobs = top_logprobs_list[0]

            # Find YES and NO logprobs (case-insensitive search)
            yes_logprob = None
            no_logprob = None
            for token, lp in token_logprobs.items():
                token_upper = token.strip().upper()
                if token_upper == "YES" and yes_logprob is None:
                    yes_logprob = lp
                elif token_upper == "NO" and no_logprob is None:
                    no_logprob = lp

            # If either token not in top-k, assign a very low logprob
            if yes_logprob is None:
                yes_logprob = -100.0
            if no_logprob is None:
                no_logprob = -100.0

            # Convert logprobs to probabilities: exp(lp) and normalize
            exp_yes = math.exp(yes_logprob)
            exp_no = math.exp(no_logprob)
            total = exp_yes + exp_no

            if total > 0:
                prob_yes = exp_yes / total
            else:
                prob_yes = 0.5  # shouldn't happen

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Error parsing logprobs, falling back to text: {e}")
            generated = result.get("choices", [{}])[0].get("text", "").strip().upper()
            prob_yes = 1.0 if generated.startswith("YES") else 0.0

        return prob_yes, latency_ms

    async def classify_json(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Send a prompt and parse the JSON response.

        Used for category/sentiment classification (production flow).

        Returns:
            (parsed_json_dict_or_None, latency_ms)
        """
        start = time.perf_counter()

        result = await self.complete(
            prompt=prompt,
            max_tokens=max_tokens or config.VLLM_MAX_TOKENS,
            temperature=0.0,
        )

        latency_ms = (time.perf_counter() - start) * 1000.0

        try:
            generated_text = result["choices"][0]["text"].strip()
            parsed = json.loads(generated_text)
            return parsed, latency_ms
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse JSON from LLM: {e}")
            # Try to extract JSON from within the text
            generated_text = result.get("choices", [{}])[0].get("text", "")
            json_match = re.search(r"\{[^}]+\}", generated_text)
            if json_match:
                try:
                    return json.loads(json_match.group()), latency_ms
                except json.JSONDecodeError:
                    pass
            return None, latency_ms

    async def get_vishing_probability(self, prompt: str) -> Tuple[float, float]:
        """
        Get a vishing probability score from the LLM (JSON output).

        Expects the LLM to return: {"vishing_probability": float}

        Returns:
            (probability, latency_ms)
        """
        start = time.perf_counter()

        result = await self.complete(
            prompt=prompt,
            max_tokens=64,
            temperature=0.0,
        )

        latency_ms = (time.perf_counter() - start) * 1000.0

        try:
            generated_text = result["choices"][0]["text"].strip()
            parsed = json.loads(generated_text)
            prob = float(parsed.get("vishing_probability", 0.0))
            if 0.0 <= prob <= 1.0:
                return prob, latency_ms
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        # Fallback: regex
        generated_text = result.get("choices", [{}])[0].get("text", "")
        pattern = r"vishing_probability\"?\s*[:=]\s*([01](?:\.\d+)?)"
        match = re.search(pattern, generated_text)
        if match:
            try:
                value = float(match.group(1))
                if 0.0 <= value <= 1.0:
                    return value, latency_ms
            except ValueError:
                pass

        return 0.0, latency_ms
