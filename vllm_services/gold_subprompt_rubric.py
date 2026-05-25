"""
Fixed sub-prompts for «gold eval» runs: six YES/NO questions per call, export includes empty GT_* columns
for human annotation next to model responses. Transcript is appended by the server (do not use {TRANSCRIPT} here).
"""

from __future__ import annotations

from typing import List, Tuple

# Bump when question wording or order changes (exports record this id).
RUBRIC_ID = "gold_eval_v2"

# (short_key, full prompt text) — order defines subprompt_1..N and gt_1..N.
# Focus text only; baseline4.build_custom_explain_chain_turn adds transcript and Greek wrappers.
SUBPROMPT_SPECS: Tuple[Tuple[str, str], ...] = (
    (
        "overall_vishing",
        "You are a vishing detection expert.\n"
        "Question: Does the conversation indicate possible vishing?\n"
        "Reply with one word only: YES or NO. If it is not clear, reply NO.",
    ),
    (
        "pressure_tactics",
        "Question: Are there pressure tactics in this conversation?\n"
        "Reply with one word only: YES or NO. If it is not clear, reply NO.",
    ),
    (
        "authority_impersonation",
        "Question: Is there authority impersonation in this conversation?\n"
        "Reply with one word only: YES or NO. If it is not clear, reply NO.",
    ),
    (
        "sensitive_info",
        "Question: Is there an attempt to extract sensitive information in this conversation?\n"
        "Reply with one word only: YES or NO. If it is not clear, reply NO.",
    ),
    (
        "trust_manipulation",
        "Question: Is there trust-building manipulation in this conversation?\n"
        "Reply with one word only: YES or NO. If it is not clear, reply NO.",
    ),
    (
        "suspicious_flow",
        "Question: Is the conversation flow suspicious for fraud?\n"
        "Reply with one word only: YES or NO. If it is not clear, reply NO.",
    ),
)

SUBPROMPT_KEYS: Tuple[str, ...] = tuple(s[0] for s in SUBPROMPT_SPECS)
DEFAULT_SUBPROMPTS: Tuple[str, ...] = tuple(s[1] for s in SUBPROMPT_SPECS)
GOLD_SUBPROMPT_COUNT: int = len(DEFAULT_SUBPROMPTS)


def subprompts_for_api() -> List[str]:
    return list(DEFAULT_SUBPROMPTS)
