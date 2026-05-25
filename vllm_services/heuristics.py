"""
Greek keyword-based heuristic scoring for vishing detection.

Provides fraud-indicator scoring based on keyword presence and contextual
combinations (urgency + sensitive data, bank + link, card + confirmation).

Ported from experiment1_mistral.py.
"""

from typing import Dict, List

from . import config


# --- Keyword weights for fraud-related indicators (Greek + Latin) ---

KEYWORD_WEIGHTS: Dict[str, float] = {
    # Strong signals — banking credentials / IDs
    "iban": 0.3,
    "pin": 0.4,
    "πιν": 0.4,
    "otp": 0.35,
    "cvv": 0.5,
    "αφμ": 0.4,
    "αμκα": 0.4,
    "κωδικός": 0.3,
    "κωδικοί": 0.3,
    "κωδικός πρόσβασης": 0.3,
    "κωδικός μιας χρήσης": 0.35,
    "επιβεβαιωτικός κωδικός": 0.35,
    "username": 0.25,
    "password": 0.35,
    # Impersonation / authority cues
    "τράπεζα": 0.2,
    "αστυνομία": 0.25,
    "δίωξη": 0.25,
    "υπουργείο": 0.2,
    "νομικής συμμόρφωσης": 0.25,
    "ασφάλειας": 0.2,
    # Threats / account actions
    "πάγω": 0.25,
    "μπλοκ": 0.25,
    "αναστολ": 0.25,
    "ακύρωση": 0.2,
    # Urgency
    "άμεσα": 0.2,
    "άμεση": 0.2,
    "επείγον": 0.25,
    "τώρα": 0.2,
    # Playbook-style vishing (bank alert / IT / courier / suspension)
    "ύποπτ": 0.22,
    "χρέωση": 0.18,
    "πάτα το ένα": 0.2,
    "πάτησε το": 0.18,
    "προσωρινός κωδικός": 0.28,
    "προσωρινό κωδικό": 0.28,
    "διαρροή": 0.2,
    "κυβερνοασφάλεια": 0.2,
    "αποτυχημένη παράδοση": 0.26,
    "επιστροφή αποστολέα": 0.22,
    "δέμα": 0.12,
    "πακέτο": 0.1,
    "ταχυμεταφορά": 0.15,
    "παράνομη δραστηριότητα": 0.28,
    "ορίστικη διαγραφή": 0.24,
    "προσωρινή αναστολή": 0.22,
    # Light FN-help cues (modest weights)
    "λόγους ασφαλείας": 0.12,
    "λόγο ασφαλείας": 0.12,
    "πατήστε το 1": 0.14,
    "πατήστε το ένα": 0.14,
    "μη εξουσιοδοτημέν": 0.14,
    "μη εξουσιοδοτημένη": 0.14,
}

URGENCY_TERMS = [
    "άμεσα",
    "άμεση",
    "αμέσως",
    "τώρα",
    "το συντομότερο",
    "επείγον",
    "σήμερα",
]

SENSITIVE_DATA_TERMS = [
    "iban",
    "pin",
    "πιν",
    "otp",
    "cvv",
    "αφμ",
    "αμκα",
    "κωδικός",
    "κωδικοί",
    "ταυτότητα",
    "ημερομηνία γέννησης",
    "πατρικό όνομα",
    "username",
    "password",
]

BANK_TERMS = [
    "τράπεζα",
    "τραπεζικός",
    "λογαριασμό",
    "λογαριασμός",
    "bank",
]

LINK_TERMS = [
    "σύνδεσμο",
    "σύνδεσμος",
    "link",
    "url",
    "ιστοσελίδα",
    "site",
]

CARD_TERMS = [
    "πιστωτική",
    "πιστωτική κάρτα",
    "χρεωστική κάρτα",
    "κάρτα",
]

CONFIRM_TERMS = [
    "επιβεβαίωση",
    "επιβεβαιώσετε",
    "επαλήθευση",
    "επιβεβαιώνω",
    "επικαιροποίηση",
]


def compute_heuristic_score(text: str) -> float:
    """
    Compute a heuristic vishing score in [0, 1] based on fraud-related keywords
    and contextual combinations.

    Scoring:
    - Sum keyword weights for each keyword present (capped at 1.0).
    - Add contextual boosts (reduced) to avoid over-scoring benign calls:
        * urgency + sensitive data
        * bank + link
        * card + confirmation
    """
    lowered = text.lower()

    # Base weighted score
    base = 0.0
    for kw, w in KEYWORD_WEIGHTS.items():
        if kw in lowered:
            base += w
    base = min(1.0, base)

    # Contextual boosts
    has_urgency = any(term in lowered for term in URGENCY_TERMS)
    has_sensitive = any(term in lowered for term in SENSITIVE_DATA_TERMS)
    has_bank = any(term in lowered for term in BANK_TERMS)
    has_link = any(term in lowered for term in LINK_TERMS)
    has_card = any(term in lowered for term in CARD_TERMS)
    has_confirm = any(term in lowered for term in CONFIRM_TERMS)

    boost = 0.0
    if has_urgency and has_sensitive:
        boost += 0.25
    if has_bank and has_link:
        boost += 0.25
    if has_card and has_confirm:
        boost += 0.25

    return min(1.0, base + boost)


def combine_scores(llm_prob: float, heuristic_score: float) -> float:
    """
    Combine LLM probability and heuristic score into a final vishing score.

    final = (1 - KEYWORD_WEIGHT) * llm_prob + KEYWORD_WEIGHT * heuristic_score
    """
    kw = config.KEYWORD_WEIGHT
    return (1.0 - kw) * llm_prob + kw * heuristic_score


def min_max_normalize(scores: List[float]) -> List[float]:
    """
    Min-max normalize scores into [0, 1]. If variance is 0, return unchanged.
    """
    if not scores:
        return scores
    s_min = min(scores)
    s_max = max(scores)
    if s_max <= s_min:
        return scores
    scale = s_max - s_min
    return [(s - s_min) / scale for s in scores]
