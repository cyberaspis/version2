"""
Centralized configuration for vllm_services.

All settings are loaded from environment variables with sensible defaults.
Use config.dev.env / config.prod.env to override per environment.
"""

import os


# --- Quality Service (optional — set to enable test run reporting) ---
QUALITY_SERVICE_URL = os.getenv("QUALITY_SERVICE_URL", "")

# --- vLLM Server (must match run_vllm.sh — default port 8010, not 8000) ---
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8010")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

# --- Classifier Server ---
CLASSIFIER_PORT = int(os.getenv("CLASSIFIER_PORT", "8001"))

# --- Automated Actions ---
ACTIONS_API_BASE_URL = os.getenv("ACTIONS_API_BASE_URL", "http://localhost:8000")
ENABLE_ACTIONS_DEFAULT = os.getenv("ENABLE_ACTIONS_DEFAULT", "true").lower() in ("1", "true", "yes")
DEBOUNCE_SECONDS = float(os.getenv("DEBOUNCE_SECONDS", "5.0"))
MIN_WORDS_FOR_CLASSIFICATION = int(os.getenv("MIN_WORDS_FOR_CLASSIFICATION", "10"))

# --- Vishing decision thresholds (progressive call scoring) ---
# Final vishing if score >= this; also stop re-classifying the call when reached.
FINAL_VISHING_STOP_THRESHOLD = float(os.getenv("FINAL_VISHING_STOP_THRESHOLD", "0.52"))
VISHING_THRESHOLD = float(os.getenv("VISHING_THRESHOLD", str(FINAL_VISHING_STOP_THRESHOLD)))

# Below this (after resolution) => SAFE; between this and FINAL => CRITICAL (elevated / ambiguous).
SAFE_SCORE_MAX = float(os.getenv("SAFE_SCORE_MAX", "0.28"))
# Legacy name kept for imports; interpreted as upper bound for SAFE in risk mapping
VISHING_CRITICAL_THRESHOLD = float(os.getenv("VISHING_CRITICAL_THRESHOLD", str(SAFE_SCORE_MAX)))
# Imported GT metrics: count CRITICAL as predicted-positive (fraud signal), not only VISHING.
METRICS_COUNT_CRITICAL_AS_VISHING_POSITIVE = os.getenv(
    "METRICS_COUNT_CRITICAL_AS_VISHING_POSITIVE", "true"
).lower() in ("1", "true", "yes")
# Recall: if keyword heuristic is high but LLM score stayed low, floor the probability.
# 0.25 avoids firing on bare-minimum keyword hits (~0.2) that caused many non-vishing FPs at 0.58.
RECALL_HEURISTIC_FLOOR_THRESHOLD = float(os.getenv("RECALL_HEURISTIC_FLOOR_THRESHOLD", "0.25"))
RECALL_HEURISTIC_FLOOR_MAX_LLM = float(os.getenv("RECALL_HEURISTIC_FLOOR_MAX_LLM", "0.42"))
# Just above FINAL threshold so floor still flips to VISHING without overshooting as hard as 0.58.
RECALL_HEURISTIC_FLOOR_VALUE = float(os.getenv("RECALL_HEURISTIC_FLOOR_VALUE", "0.54"))
# When the model echoes prompt examples ("Παράδειγμα", "JSON:"), cap parsed vishing_probability (reduces FP).
COMPLETION_ECHO_MAX_PARSED_PROB = float(os.getenv("COMPLETION_ECHO_MAX_PARSED_PROB", "0.24"))
# CRITICAL-band second pass: discrete triage (VISHING/SAFE/UNCERTAIN) instead of repeating the JSON probability prompt.
USE_CRITICAL_TRIAGE_SECOND_PASS = os.getenv(
    "USE_CRITICAL_TRIAGE_SECOND_PASS", "true"
).lower() in ("1", "true", "yes")
# Map triage JSON labels back to numeric scores for risk_status + UI.
# UNCERTAIN was 0.40 → stayed in CRITICAL band (0.28–0.52) while still counting as fraud-positive in metrics — confusing.
# Default 0.54 (≥ FINAL_VISHING_STOP_THRESHOLD): triage "can't decide" still surfaces as VISHING risk_status, aligned with policy.
CRITICAL_TRIAGE_SCORE_VISHING = float(os.getenv("CRITICAL_TRIAGE_SCORE_VISHING", "0.58"))
CRITICAL_TRIAGE_SCORE_SAFE = float(os.getenv("CRITICAL_TRIAGE_SCORE_SAFE", "0.14"))
CRITICAL_TRIAGE_SCORE_UNCERTAIN = float(os.getenv("CRITICAL_TRIAGE_SCORE_UNCERTAIN", "0.54"))
# Baseline: extra full-context LLM pass when first pass is SAFE but heuristics are warm.
BASELINE_SAFE_RECHECK_HEURISTIC_MIN = float(os.getenv("BASELINE_SAFE_RECHECK_HEURISTIC_MIN", "0.20"))
BASELINE_SAFE_RECHECK_PROB_BELOW = float(os.getenv("BASELINE_SAFE_RECHECK_PROB_BELOW", "0.42"))
# When the first pass stays SAFE but domain is fraud-prone (bank/utilities/education/verification),
# run the same discrete triage used in the CRITICAL band (many FNs are ~0–0.2 in these domains).
USE_SAFE_DOMAIN_TRIAGE_RECHECK = os.getenv(
    "USE_SAFE_DOMAIN_TRIAGE_RECHECK", "true"
).lower() in ("1", "true", "yes")
SAFE_DOMAIN_TRIAGE_MAX_PROB = float(os.getenv("SAFE_DOMAIN_TRIAGE_MAX_PROB", "0.42"))
_safe_dom_raw = os.getenv(
    "SAFE_DOMAIN_TRIAGE_DOMAINS",
    "Banking,Utilities,Account Verification,Education",
)
SAFE_DOMAIN_TRIAGE_DOMAINS = frozenset(
    x.strip() for x in _safe_dom_raw.split(",") if x.strip()
)
# Live classify: long transcripts + very SAFE + quiet heuristics → second full-context pass (reduces FN on subtle vishing).
CLASSIFY_SAFE_LONG_RECHECK_ENABLED = os.getenv(
    "CLASSIFY_SAFE_LONG_RECHECK_ENABLED", "true"
).lower() in ("1", "true", "yes")
CLASSIFY_SAFE_RECHECK_MIN_WORDS = int(os.getenv("CLASSIFY_SAFE_RECHECK_MIN_WORDS", "220"))
CLASSIFY_SAFE_RECHECK_MAX_HEURISTIC = float(
    os.getenv("CLASSIFY_SAFE_RECHECK_MAX_HEURISTIC", "0.12")
)
CLASSIFY_SAFE_RECHECK_PROB_BELOW = float(os.getenv("CLASSIFY_SAFE_RECHECK_PROB_BELOW", "0.22"))

# --- Progressive vishing probability ---
# Starting score when a call has no prior classification.
STARTING_VISHING_PROBABILITY = float(os.getenv("STARTING_VISHING_PROBABILITY", "0.0"))
ACCUMULATE_VISHING_PROB = os.getenv("ACCUMULATE_VISHING_PROB", "true").lower() in ("1", "true", "yes")
# Blend prior with new LLM estimate (allows score to go down when conversation turns normal).
PROGRESS_LLM_WEIGHT = float(os.getenv("PROGRESS_LLM_WEIGHT", "0.62"))
# Extra bump from fraud-like keywords in the *latest segment* (pressure, OTP, etc.), scaled and capped.
SEGMENT_HEURISTIC_SCALE = float(os.getenv("SEGMENT_HEURISTIC_SCALE", "0.28"))
SEGMENT_HEURISTIC_CAP = float(os.getenv("SEGMENT_HEURISTIC_CAP", "0.22"))
# When full transcript heuristic is strong, nudge probability up (vishing recall).
FULL_HEURISTIC_NUDGE_THRESHOLD = float(os.getenv("FULL_HEURISTIC_NUDGE_THRESHOLD", "0.16"))
FULL_HEURISTIC_NUDGE_AMOUNT = float(os.getenv("FULL_HEURISTIC_NUDGE_AMOUNT", "0.20"))

# Ambiguous band (~50%): nudge toward VISHING or SAFE using heuristics if the score sits here.
NEAR_FIFTY_LOW = float(os.getenv("NEAR_FIFTY_LOW", "0.45"))
NEAR_FIFTY_HIGH = float(os.getenv("NEAR_FIFTY_HIGH", "0.55"))
AMBIGUOUS_NUDGE_UP = float(os.getenv("AMBIGUOUS_NUDGE_UP", "0.12"))
AMBIGUOUS_NUDGE_DOWN = float(os.getenv("AMBIGUOUS_NUDGE_DOWN", "0.10"))
# If latest segment heuristic >= this, nudge up out of the band; if both low, nudge down.
AMBIGUOUS_PRESSURE_SEG = float(os.getenv("AMBIGUOUS_PRESSURE_SEG", "0.12"))
AMBIGUOUS_CLEAN_SEG = float(os.getenv("AMBIGUOUS_CLEAN_SEG", "0.05"))
AMBIGUOUS_CLEAN_FULL = float(os.getenv("AMBIGUOUS_CLEAN_FULL", "0.10"))

# --- vLLM Client ---
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "120.0"))
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "256"))
# Must match (or stay below) the vLLM server --max-model-len; used to cap classifier prompts.
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "12288"))
# Behavior-classifier JSON completion. Prompt + completion must fit in VLLM_MAX_MODEL_LEN (one shared budget).
# Lower value => higher allowed prompt size (e.g. 6144 - 64 = 6080 max prompt tokens from vLLM).
VLLM_CLASSIFIER_COMPLETION_TOKENS = int(os.getenv("VLLM_CLASSIFIER_COMPLETION_TOKENS", "64"))
# CRITICAL triage returns a tiny JSON; lower max_tokens = faster decode on vLLM.
VLLM_TRIAGE_COMPLETION_TOKENS = int(os.getenv("VLLM_TRIAGE_COMPLETION_TOKENS", "48"))
# Explainability: Prompt #4 split into one vLLM call per sub-question, then normal classifier unchanged.
# Short Greek explainability per sub-question; raise if outputs truncate mid-sentence.
VLLM_EXPLAINABILITY_STEP_TOKENS = int(os.getenv("VLLM_EXPLAINABILITY_STEP_TOKENS", "260"))
VLLM_EXPLAINABILITY_FINAL_TOKENS = int(os.getenv("VLLM_EXPLAINABILITY_FINAL_TOKENS", "480"))
# Custom explainability modal (full prompt + chained sub-prompts): caps per vLLM call.
# Sub-prompt completions use MAX_OUTPUT_TOKENS_PER_CALL; custom full prompt uses MAIN_PROMPT_MAX_NEW_TOKENS
# capped by MAX_OUTPUT_TOKENS_MAIN_PROMPT so you can keep subs short (e.g. ΝΑΙ/ΟΧΙ) while allowing longer full answers.
MAIN_PROMPT_MAX_NEW_TOKENS = max(1, int(os.getenv("MAIN_PROMPT_MAX_NEW_TOKENS", "512")))
SUBPROMPT_MAX_NEW_TOKENS = max(1, int(os.getenv("SUBPROMPT_MAX_NEW_TOKENS", "24")))
# Packing transcript + instructions per call (also bounded by VLLM_MAX_MODEL_LEN − completion − guard).
# With max_model_len=6144, ~5500 leaves room for ~24 completion tokens + guard.
MAX_INPUT_TOKENS_PER_CALL = max(256, int(os.getenv("MAX_INPUT_TOKENS_PER_CALL", "5500")))
MAX_OUTPUT_TOKENS_PER_CALL = max(1, int(os.getenv("MAX_OUTPUT_TOKENS_PER_CALL", "24")))
MAX_OUTPUT_TOKENS_MAIN_PROMPT = max(1, int(os.getenv("MAX_OUTPUT_TOKENS_MAIN_PROMPT", "512")))
# Chained custom sub-prompts: prose (2–4 sentences) or yesno (only ΝΑΙ or ΟΧΙ) — see baseline4.build_custom_explain_chain_turn.
CUSTOM_EXPLAIN_SUB_STYLE = os.getenv("CUSTOM_EXPLAIN_SUB_STYLE", "yesno").strip().lower()
# Custom explainability modal: max chained sub-prompts per call (each → one Mistral completion).
CUSTOM_EXPLAIN_MAX_SUBPROMPTS = max(1, int(os.getenv("CUSTOM_EXPLAIN_MAX_SUBPROMPTS", "12")))
# Extra slack under (MAX_LEN - completion) when using the HF tokenizer count
# (vLLM may add specials; chat fallback adds template tokens — keep headroom).
CLASSIFIER_PROMPT_TOKEN_GUARD = int(os.getenv("CLASSIFIER_PROMPT_TOKEN_GUARD", "128"))
# Legacy: char-based fallback only; prefer tokenizer in baseline4.
CLASSIFIER_PROMPT_SAFETY_TOKENS = int(os.getenv("CLASSIFIER_PROMPT_SAFETY_TOKENS", "384"))
# Transcript packing for long calls: head + N windows from the middle + tail (scam can sit in the gap).
CLASSIFIER_HEAD_WORDS = int(os.getenv("CLASSIFIER_HEAD_WORDS", "160"))
CLASSIFIER_TAIL_WORDS = int(os.getenv("CLASSIFIER_TAIL_WORDS", "260"))
# Words per middle window; CLASSIFIER_MID_WINDOWS chunks spread across the skipped region (1/(N+1), 2/(N+1), …).
CLASSIFIER_MID_WORDS = int(os.getenv("CLASSIFIER_MID_WORDS", "160"))
CLASSIFIER_MID_WINDOWS = int(os.getenv("CLASSIFIER_MID_WINDOWS", "2"))
CLASSIFIER_MAX_PROMPT_WORDS_FULL = int(os.getenv("CLASSIFIER_MAX_PROMPT_WORDS_FULL", "880"))
# Baseline / import replay: run one LLM step per transcript segment (cumulative text), not one shot on full join.
# Default false: one LLM pass per call (head+tail) — much faster than per-segment replay.
# Set USE_SEGMENT_PROGRESSIVE_BASELINE=true for slower, segment-by-segment baseline.
USE_SEGMENT_PROGRESSIVE_BASELINE = os.getenv("USE_SEGMENT_PROGRESSIVE_BASELINE", "false").lower() in (
    "1",
    "true",
    "yes",
)
# Live classify: segment-by-segment = many LLM calls per long call (slow). Default single-shot (fast).
USE_SEGMENT_PROGRESSIVE_CLASSIFY = os.getenv("USE_SEGMENT_PROGRESSIVE_CLASSIFY", "false").lower() in (
    "1",
    "true",
    "yes",
)
# When progressive classify is on, only use it if segment count ≤ this (else one LLM call).
CLASSIFY_MAX_SEGMENTS_FOR_PROGRESSIVE = int(os.getenv("CLASSIFY_MAX_SEGMENTS_FOR_PROGRESSIVE", "12"))
# How many calls to classify in parallel against vLLM (batch / SSE run). Use 1 for throughput/latency checks; 100 for speed.
BASELINE_BATCH_CONCURRENCY = max(1, int(os.getenv("BASELINE_BATCH_CONCURRENCY", "100")))
# Upper bound for concurrency (safety). Override with BASELINE_BATCH_CONCURRENCY_CAP if needed.
BASELINE_BATCH_CONCURRENCY_CAP = max(1, int(os.getenv("BASELINE_BATCH_CONCURRENCY_CAP", "256")))
VLLM_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.0"))
VLLM_LOGPROBS = int(os.getenv("VLLM_LOGPROBS", "10"))

# --- Short, zero-keyword calls (greetings, trivial chit-chat) ---
# When the transcript is very short AND keyword heuristics stay near zero, cap the
# numeric score so recall-oriented prompts do not label benign one-liners as vishing.
BENIGN_SHORT_MAX_WORDS = int(os.getenv("BENIGN_SHORT_MAX_WORDS", "48"))
BENIGN_SHORT_MAX_HEURISTIC = float(os.getenv("BENIGN_SHORT_MAX_HEURISTIC", "0.06"))
BENIGN_SHORT_SCORE_CAP = float(os.getenv("BENIGN_SHORT_SCORE_CAP", "0.26"))
BENIGN_SHORT_PROMPT_NOTE = os.getenv("BENIGN_SHORT_PROMPT_NOTE", "true").lower() in (
    "1",
    "true",
    "yes",
)

# --- Heuristic Scoring ---
# Combined score weight: final = (1 - KEYWORD_WEIGHT) * llm_prob + KEYWORD_WEIGHT * heuristic
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.6"))

# --- Call list broadcast (WebSocket) ---
# Include calls whose last segment or last classification is newer than this (seconds).
CALL_LIST_MAX_AGE_SECONDS = float(os.getenv("CALL_LIST_MAX_AGE_SECONDS", "86400"))

# --- Environment ---
ENV = os.getenv("ENV", "dev")  # "dev" or "prod"

# --- Prompt File ---
PROMPT_FILE = os.getenv("PROMPT_FILE", "prompt.txt")
