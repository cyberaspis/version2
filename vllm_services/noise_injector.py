"""
Greek ASR noise simulation for vishing detection robustness testing.

Injects realistic transcription errors:
- Homophone replacements (Greek near-homophones)
- Word swaps
- Random deletions
- Word repetitions (ASR echo simulation)

Direct port from experiments/noise_injector.py.
"""

import random
from typing import List, Sequence


# Greek homophone / near-homophone mappings
HOMOPHONE_MAP_LIGHT = {
    "και": ["κι"],
    "κι": ["και"],
    "δεν": ["δε"],
    "δε": ["δεν"],
    "σε": ["σ'"],
    "σου": ["σου", "σ'"],
    "μου": ["μου", "μ'"],
    "είναι": ["ναι", "είναι"],
    "ναι": ["ναι", "νε"],
    "όχι": ["όχι", "οχι"],
}

HOMOPHONE_MAP_HEAVY = {
    **HOMOPHONE_MAP_LIGHT,
    "ή": ["η", "ι"],
    "η": ["ή", "ι"],
    "θα": ["θα", "δα"],
    "πως": ["πως", "πώς"],
    "πώς": ["πως", "πώσ"],
}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.split()


def _detokenize(tokens: Sequence[str]) -> str:
    """Join tokens with single spaces."""
    return " ".join(tokens)


def _replace_with_homophones(
    tokens: List[str],
    homophone_map: dict,
    prob: float,
) -> List[str]:
    """Replace some tokens with homophones based on a probability."""
    new_tokens: List[str] = []
    for tok in tokens:
        key = tok.lower()
        if key in homophone_map and random.random() < prob:
            candidates = homophone_map[key]
            replacement = random.choice(candidates) if candidates else tok
            # Preserve capitalization style (naive)
            if tok.istitle():
                replacement = replacement.capitalize()
            elif tok.isupper():
                replacement = replacement.upper()
            new_tokens.append(replacement)
        else:
            new_tokens.append(tok)
    return new_tokens


def _random_swaps(tokens: List[str], swap_prob: float) -> List[str]:
    """Randomly swap neighboring tokens."""
    tokens = tokens[:]
    i = 0
    while i < len(tokens) - 1:
        if random.random() < swap_prob:
            tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
            i += 2
        else:
            i += 1
    return tokens


def _random_deletions(tokens: List[str], delete_prob: float) -> List[str]:
    """Randomly delete tokens."""
    return [tok for tok in tokens if random.random() >= delete_prob]


def _random_repetitions(tokens: List[str], repeat_prob: float) -> List[str]:
    """Randomly repeat tokens once (ASR echo simulation)."""
    new_tokens: List[str] = []
    for tok in tokens:
        new_tokens.append(tok)
        if random.random() < repeat_prob:
            new_tokens.append(tok)
    return new_tokens


def add_light_noise(text: str) -> str:
    """
    Add light noise: low-probability homophone replacements + occasional word swaps.
    """
    tokens = _tokenize(text)
    if not tokens:
        return text
    tokens = _replace_with_homophones(tokens, HOMOPHONE_MAP_LIGHT, prob=0.15)
    tokens = _random_swaps(tokens, swap_prob=0.05)
    return _detokenize(tokens)


def add_heavy_noise(text: str) -> str:
    """
    Add heavy noise: aggressive homophones + deletions + repetitions.
    """
    tokens = _tokenize(text)
    if not tokens:
        return text
    tokens = _replace_with_homophones(tokens, HOMOPHONE_MAP_HEAVY, prob=0.4)
    tokens = _random_deletions(tokens, delete_prob=0.15)
    tokens = _random_repetitions(tokens, repeat_prob=0.1)
    return _detokenize(tokens)


def apply_noise(texts: Sequence[str], level: str) -> List[str]:
    """
    Apply noise to a list of texts.

    Args:
        texts: input strings
        level: "light" or "heavy"
    """
    if level not in {"light", "heavy"}:
        raise ValueError('level must be either "light" or "heavy"')

    noisy: List[str] = []
    for text in texts:
        if level == "light":
            noisy.append(add_light_noise(text))
        else:
            noisy.append(add_heavy_noise(text))
    return noisy
