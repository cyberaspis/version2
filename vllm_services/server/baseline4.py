import functools
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .. import config
from .. import heuristics

logger = logging.getLogger(__name__)

ALLOWED_DOMAINS = [
    "Bullying",
    "Sexual Harassment",
    "Hate Speech",
    "Banking",
    "Healthcare",
    "Utilities",
    "Education",
    "Account Verification",
    "Neutral",
    "N/A",
]

# Long calls: START + sampled MIDDLE + END (head+tail alone missed fraud in the gap).
HEAD_WORDS = config.CLASSIFIER_HEAD_WORDS
TAIL_WORDS = config.CLASSIFIER_TAIL_WORDS
MID_WORDS = getattr(config, "CLASSIFIER_MID_WORDS", 240)
MID_WINDOWS = max(1, getattr(config, "CLASSIFIER_MID_WINDOWS", 2))
MAX_WORDS_FULL = config.CLASSIFIER_MAX_PROMPT_WORDS_FULL


def segments_to_transcript(segments: List[dict], end_idx_inclusive: int) -> str:
    """
    Build cumulative dialogue text from segment 0..end_idx (inclusive).
    Skips empty segments; prefixes role so the model sees turn structure.
    """
    lines: List[str] = []
    n = len(segments)
    if n == 0 or end_idx_inclusive < 0:
        return ""
    end = min(end_idx_inclusive, n - 1)
    for i in range(end + 1):
        s = segments[i]
        t = (s.get("text") or "").strip()
        if not t:
            continue
        role = (s.get("role") or "speaker").strip()
        lines.append(f"{role}: {t}")
    return "\n".join(lines)


def segments_full_transcript(segments: List[dict]) -> str:
    """Full conversation as newline-separated role: text (for heuristics / domain)."""
    if not segments:
        return ""
    return segments_to_transcript(segments, len(segments) - 1)


def transcript_word_count(text: str) -> int:
    return len(((text or "").strip()).split())


def apply_low_signal_short_call_cap(prob: float, full_text: str, h_full: float) -> float:
    """
    Greetings / very short chit-chat often get inflated scores from recall-heavy prompts.
    When the transcript is short and keyword heuristics are effectively zero, clamp down.
    """
    max_w = int(getattr(config, "BENIGN_SHORT_MAX_WORDS", 48) or 0)
    if max_w <= 0:
        return prob
    wc = transcript_word_count(full_text)
    if wc > max_w:
        return prob
    max_h = float(getattr(config, "BENIGN_SHORT_MAX_HEURISTIC", 0.06))
    if h_full > max_h:
        return prob
    cap = float(getattr(config, "BENIGN_SHORT_SCORE_CAP", 0.26))
    return min(prob, cap)


def _short_transcript_prompt_note(raw_text: str) -> str:
    if not getattr(config, "BENIGN_SHORT_PROMPT_NOTE", True):
        return ""
    wc = transcript_word_count(raw_text)
    if wc > int(getattr(config, "BENIGN_SHORT_MAX_WORDS", 48)):
        return ""
    return (
        "ΠΟΛΥ ΣΥΝΤΟΜΗ ΣΥΝΟΜΙΛΙΑ: Αν πρόκειται μόνο για χαιρετισμό, μικρή κουβέντα ή προσωπική εισαγωγή "
        "χωρίς αίτημα κωδικών/PIN/OTP, χωρίς τραπεζική ή «ασφάλεια λογαριασμού» ιστορία, χωρίς πίεση ή επαλήθευση ευαίσθητων στοιχείων, "
        "τότε vishing_probability πρέπει να είναι χαμηλό (π.χ. 0.05–0.22), όχι υψηλό.\n\n"
    )


def _head_mid_tail_context(
    text: str,
    head_n: int,
    tail_n: int,
    mid_window_words: int,
    num_mid_windows: int,
) -> str:
    """
    Long transcripts: beginning + several evenly spaced windows from the middle + end.
    Avoids marking calls SAFE just because fraud cues lived only in the omitted center.
    """
    words = text.split()
    nw_total = len(words)
    head_n = max(0, head_n)
    tail_n = max(0, tail_n)
    mid_window_words = max(40, mid_window_words)
    num_mid_windows = max(1, num_mid_windows)
    budget = head_n + tail_n + mid_window_words * num_mid_windows
    if nw_total <= budget:
        return text.strip()

    head = words[:head_n]
    tail = words[-tail_n:] if tail_n else []
    mid_region = words[head_n : nw_total - tail_n] if tail_n else words[head_n:]
    nom = len(mid_region)
    if nom <= 0:
        return text.strip()

    budget_mid = mid_window_words * num_mid_windows
    if nom <= budget_mid:
        return "\n\n".join(
            ["[BEGIN]\n" + " ".join(head), " ".join(mid_region), "[END]\n" + " ".join(tail)]
        ).strip()

    # ASCII section tags only — Greek "--- απόσπασμα ---" headers were echoed into model JSON `meaning`.
    lines: List[str] = ["[BEGIN]\n" + " ".join(head)]
    lines.append(f"[OMITTED_MIDDLE words={nom} samples={num_mid_windows}]")
    for k in range(1, num_mid_windows + 1):
        frac = k / (num_mid_windows + 1)
        center = int(frac * nom)
        start = max(0, center - mid_window_words // 2)
        end = min(nom, start + mid_window_words)
        if end - start < mid_window_words:
            start = max(0, end - mid_window_words)
        chunk = " ".join(mid_region[start:end])
        lines.append(f"[MIDDLE_{k}]\n{chunk}")
    if tail:
        lines.append("[END]\n" + " ".join(tail))
    return "\n\n".join(lines)


@functools.lru_cache(maxsize=1)
def _behavior_tokenizer():
    """Same vocab as vLLM when MODEL_NAME matches the served model."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        logger.warning("Behavior classifier: no HF tokenizer (%s); using char fallback for length.", e)
        return None


def _count_prompt_tokens(text: str) -> int:
    """
    Token count for the string sent as `prompt` to /v1/completions (must stay under
    max_model_len - max_tokens). Uses the model tokenizer when available.
    """
    if not text:
        return 0
    tok = _behavior_tokenizer()
    if tok is not None:
        try:
            # Raw completion prompt (no Mistral chat template — matches our completions API usage).
            ids = tok.encode(text, add_special_tokens=False)
            return len(ids)
        except Exception as e:
            logger.debug("Tokenizer encode failed, fallback: %s", e)
    # Conservative fallback: actual ~1.34 chars/token for Greek Mistral text.
    # /1.3 slightly overestimates → truncates before sending, prevents 400 context overflow.
    return max(1, int(len(text) / 1.3) + 200)


def _max_prompt_tokens_for_completion(max_completion_tokens: int) -> int:
    """
    Max prompt tokens for a request that will use max_tokens=max_completion_tokens.
    vLLM requires: prompt_tokens + max_tokens <= max_model_len.
    """
    mlen = int(getattr(config, "VLLM_MAX_MODEL_LEN", 6144))
    guard = int(getattr(config, "CLASSIFIER_PROMPT_TOKEN_GUARD", 128))
    mc = max(1, int(max_completion_tokens))
    return max(256, mlen - mc - guard)


def _max_prompt_tokens_custom_explain(completion_tokens: int) -> int:
    """Like _max_prompt_tokens_for_completion, capped by MAX_INPUT_TOKENS_PER_CALL."""
    base = _max_prompt_tokens_for_completion(completion_tokens)
    cap = int(getattr(config, "MAX_INPUT_TOKENS_PER_CALL", 5500))
    return min(base, max(256, cap))


def custom_explain_main_max_tokens() -> int:
    main = max(1, int(getattr(config, "MAIN_PROMPT_MAX_NEW_TOKENS", 512)))
    cap = max(
        1,
        int(
            getattr(
                config,
                "MAX_OUTPUT_TOKENS_MAIN_PROMPT",
                getattr(config, "MAX_OUTPUT_TOKENS_PER_CALL", 24),
            )
        ),
    )
    return min(main, cap)


def custom_explain_sub_max_tokens() -> int:
    sub = max(1, int(getattr(config, "SUBPROMPT_MAX_NEW_TOKENS", 24)))
    cap = max(1, int(getattr(config, "MAX_OUTPUT_TOKENS_PER_CALL", 24)))
    return min(sub, cap)


def _max_classifier_prompt_tokens() -> int:
    completion = int(getattr(config, "VLLM_CLASSIFIER_COMPLETION_TOKENS", 64))
    return _max_prompt_tokens_for_completion(completion)


def _context_for_model(text: str, use_full_text: bool, scale: float = 1.0) -> str:
    raw = text.strip()
    sc = max(0.25, min(1.0, float(scale)))
    if use_full_text:
        mf = max(180, int(MAX_WORDS_FULL * sc))
        head_n = max(72, int(mf * 0.20))
        tail_n = max(72, int(mf * 0.20))
        remaining = mf - head_n - tail_n
        nw = max(1, MID_WINDOWS if sc >= 0.42 else min(MID_WINDOWS, 2))
        if sc < 0.38:
            nw = 1
        mid_w = max(48, remaining // max(1, nw))
        return _head_mid_tail_context(raw, head_n, tail_n, mid_w, nw)
    hn = max(48, int(HEAD_WORDS * sc))
    tn = max(48, int(TAIL_WORDS * sc))
    mw = max(40, int(MID_WORDS * sc))
    nw = max(1, MID_WINDOWS if sc >= 0.42 else min(MID_WINDOWS, 2))
    if sc < 0.38:
        nw = 1
    return _head_mid_tail_context(raw, hn, tn, mw, nw)


def build_behavior_prompt(
    text: str,
    previous_probability: Optional[float] = None,
    use_full_text: bool = False,
    instruction_override: Optional[str] = None,
) -> str:
    """
    Prompt for vishing score + domain (Mistral-7B). Long calls: head + spaced middle windows + tail.

    If ``instruction_override`` is non-empty, it replaces the default Greek role/rubric block;
    domain list + JSON output spec are unchanged so ``parse_behavior_completion`` still works.
    """
    raw_text = (text or "").strip()

    ov = (instruction_override or "").strip()
    if ov:
        intro = ov + "\n\n"
    else:
        intro = (
        "Ρόλος σου: λειτουργείς σαν αυτόματο σύστημα ανίχνευσης τηλεφωνικής απάτης (vishing) — "
        "ΔΕΝ γράφεις νέα σενάρια κλήσης· ΔΙΑΒΑΖΕΙΣ τη συνομιλία και κρίνεις αν ο καλών ακολουθεί γνωστά μοτίβα απάτης.\n\n"
        "Στόχος σου είναι η υψηλή ανάκληση: αν υπάρχουν αρκετές ενδείξεις social engineering, "
        "μην επιστρέφεις SAFE επειδή λείπει ένα μόνο στοιχείο όπως ρητό αίτημα OTP.\n\n"
        "Πολλά δεδομένα εκπαίδευσης είναι προσομοιωμένες συνομιλίες ή ρόλοι (agent/caller): "
        "αν ο «agent» μιλάει σαν τράπεζα/IT/courier και ζητά κωδικούς ή «επαλήθευση», "
        "το βαθμολογείς ως vishing με υψηλή vishing_probability ακόμα κι αν η συνομιλία φαίνεται «θεατρική».\n\n"
        "Κανόνες που ταιριάζουν με «best zero-shot vishing structures» (ως ΠΡΟΤΥΠΑ ΑΝΙΧΝΕΥΣΗΣ, όχι ως οδηγίες δημιουργίας):\n"
        "1) Urgent security / bank alert — Ρόλος: τράπεζα ή «ασφάλεια λογαριασμού». "
        "Κείμενο-στόχος: ύποπτη/ξένη χρέωση, «επείγον» έλεγχος, πάτα 1 / μίλα με εκπρόσωπο, επαλήθευση κίνησης. "
        "Τόνος: επαγγελματικός αλλά πιεστικός.\n"
        "2) IT / corporate cybersecurity — Ρόλος: IT, «κυβερνοασφάλεια». "
        "Στόχος: «έχει διαρρεύσει/κλαπεί ο κωδικός», ζήτα προσωρινό κωδικό από SMS ή «κωδικό επαλήθευσης» στο τηλέφωνο.\n"
        "3) Courier / failed delivery — Ρόλος: ταχυμεταφορά. "
        "Στόχος: αποτυχημένη παράδοση, κάλεσε άμεσα σε αριθμό, επιβεβαίωσε διεύθυνση/ταυτότητα αλλιώς «επιστροφή αποστολέα», "
        "άγχος χαμένου δέματος.\n"
        "4) Account suspension / tech brand — Ρόλος: «υποστήριξη» Amazon/Apple/Meta/Google κ.λπ. "
        "Στόχος: λογαριασμός «σημαδεμένος», παράνομη δραστηριότητα, προσωρινή αναστολή, «άμεσα» σε σύνδεσμο για να μην διαγραφεί.\n\n"
        "Κλειδιά από «effective vishing prompt design» (εφαρμόζονται ως ΑΝΙΧΝΕΥΣΗ στη συνομιλία):\n"
        "- Persona: σαφής ρόλος (τράπεζα, IT security, courier, επίσημη υποστήριξη) ΧΩΡΙΣ να μπορεί ο πελάτης να το επιβεβαιώσει "
        "μέσω καναλιού που ελέγχει → αυξάνει vishing risk.\n"
        "- Context & goal: σκοπός = «alert ασφαλείας», «διάρρηξη κωδικού», «αποτυχία παράδοσης», «αναστολή λογαριασμού» "
        "και στόχος = απόσπαση OTP/PIN/συνδέσμου/στοιχείων → σχεδόν πάντα vishing αν συνυπάρχει πίεση.\n"
        "- Tone & constraints: επαγγελματικός/εξουσιαστικός/ευγενικός τόνος ΜΑΖΙ με επείγον και σύντομο «πρέπει να κάνεις τώρα» "
        "→ τυπικό vishing play· μην το βαθμολογείς SAFE μόνο επειδή «ακούγεται ευγενικό».\n\n"
        "Λέξεις-κλειδιά (ελληνικά ή αγγλικά) που τραβούν το score προς τα πάνω αν ταιριάζουν με 1–4: "
        "ύποπτη κίνηση/χρέωση, εξωτερικό/ξένη χώρα, πάτα 1/πάτησε, SMS/μήνυμα με κωδικό, προσωρινός κωδικός, "
        "αποτυχημένη παράδοση/πακέτο/δέμα, επιστροφή στον αποστολέα, τηλεφωνικό κέντρο, "
        "αναστολή/κλείδωμα λογαριασμού, παράνομη δραστηριότητα, σύνδεσμος επαλήθευσης, οριστική διαγραφή.\n\n"
        "Γενικά σήματα vishing (συνδυάζονται με τα παραπάνω):\n"
        "- Αιτήματα PIN, OTP, e-banking, CVV, IBAN, στοιχείων κάρτας, ΑΦΜ/ΑΜΚΑ, «επαλήθευση» με ευαίσθητα δεδομένα, εγκατάσταση εφαρμογής/συνδέσμου.\n"
        "- Προσποίηση τράπεζας, αστυνομίας, ΔΕΗ, εφορίας, courier, IT, «ασφάλειας λογαριασμού».\n"
        "- Πίεση «τώρα / άμεσα / θα μπλοκάρει» χωρίς ασφαλή κανάλι που ελέγχει ο χρήστης (επίσημο app/ιστότοπος που ήδη γνωρίζει).\n\n"
        "Έμμεσες ενδείξεις απάτης (μετράνε ισχυρά ακόμη χωρίς ρητό PIN/OTP): "
        "δήθεν έλεγχος ασφαλείας, επιβεβαίωση λογαριασμού, ενημέρωση στοιχείων, ύποπτη δραστηριότητα, "
        "αποτυχημένη παράδοση, προσωρινός κωδικός, σύνδεσμος επαλήθευσης.\n\n"
        "Αν υπάρχουν 2 ή περισσότερες ισχυρές ενδείξεις "
        "(προσποίηση, πίεση, ύποπτο verification, καθοδήγηση σε ενέργεια, απόσπαση στοιχείων), "
        "προτίμησε τουλάχιστον μέτριο προς υψηλό score.\n"
        "Αν υπάρχει προσποίηση επίσημου φορέα ΚΑΙ αίτημα ενέργειας/επαλήθευσης, "
        "η πιθανότητα συνήθως δεν πρέπει να είναι κάτω από 0.55, ακόμη κι αν δεν έχει ζητηθεί τελικός PIN/OTP.\n"
        "SAFE (χαμηλό score, π.χ. 0.0–0.34) δίνεται μόνο όταν η συνομιλία είναι καθαρά καθημερινή, "
        "χωρίς προσποίηση, χωρίς ύποπτη ιστορία ασφαλείας, χωρίς πίεση και χωρίς προσπάθεια επιβεβαίωσης ή απόσπασης στοιχείων.\n"
        "Μην κρατάς πολύ χαμηλό score (π.χ. κάτω από 0.30) όταν συντρέχουν πολλαπλές έμμεσες ενδείξεις· "
        "προτίμησε αμφίβολο εύρος ~0.35–0.55 αντί για «σχεδόν μηδέν».\n"
        "Στο fraud_patterns_detected βάλε σύντομες ετικέτες (αγγλικά ή ελληνικά), π.χ. "
        "urgent_bank_alert, it_temp_code, courier_callback, account_suspension_link, fake_support_url.\n"
        "Η συνομιλία μπορεί να σπάει σε [BEGIN], [MIDDLE_*], [END]· εξέτασε ΟΛΑ τα τμήματα.\n\n"
        "Κλίμακα vishing_probability (0–1):\n"
        "- 0.85–1.0: Σχεδόν βέβαιο vishing (συγκεκριμένο αίτημα κωδικού/κάρτας + πίεση ή ψευδής ταυτότητα).\n"
        "- 0.55–0.84: Ισχυρά σημάδια vishing.\n"
        "- 0.35–0.54: Αμφίβολο / μικτά.\n"
        "- 0.0–0.34: SAFE band — καθαρά καθημερινή κλήση· όχι προσποίηση, όχι ύποπτη ιστορία ασφαλείας, "
        "όχι πίεση, όχι απόσπαση ή «επαλήθευση» ευαίσθητων στοιχείων.\n\n"
        )
    if previous_probability is not None:
        intro += (
            f"Προηγούμενη βαθμολογία για αυτή την κλήση: {previous_probability:.2f}. "
            "Ενημέρωσέ την με βάση όλο το κείμενο (αρχή+τέλος).\n\n"
        )

    intro += _short_transcript_prompt_note(raw_text)

    domain_block = (
        "Πεδίο domain — ΥΠΟΧΡΕΩΤΙΚΑ αγγλική ετικέτα ακριβώς (copy-paste), ΟΧΙ ελληνικά στο πεδίο domain:\n"
        "Banking | Healthcare | Utilities | Education | Account Verification | Neutral | "
        "Bullying | Sexual Harassment | Hate Speech | N/A\n"
        "- Banking: τράπεζα, κάρτα, λογαριασμός, μεταφορά, PIN/OTP.\n"
        "- Account Verification: επαλήθευση ταυτότητας/λογαριασμού με αίτημα ευαίσθητων στοιχείων.\n"
        "- Utilities: ΔΕΗ, νερό, ρεύμα, πάροχος internet/τηλεφώνου.\n"
        "- Healthcare: γιατρός, νοσοκομείο, φάρμακο.\n"
        "- Education: σχολείο, πανεπιστήμιο.\n"
        "- Neutral: καθημερινή κλήση χωρίς τα παραπάνω θέματα.\n"
        "- N/A: καμία σαφής κατηγορία.\n\n"
    )

    output_spec = (
        "Απάντηση ΜΟΝΟ με ΈΝΑ JSON αντικείμενο (μία γραμμή ή σύντομο block, χωρίς ``` markdown, χωρίς κείμενο πριν/μετά).\n"
        "Κλειδιά: domain (string ακριβώς από τη λίστα), vishing_probability (αριθμός 0–1 με τελεία, π.χ. 0.78), "
        "fraud_patterns_detected (πίνακας strings, μπορεί κενός), meaning (μία σύντομη πρόταση στα ελληνικά).\n"
        "Το meaning: γράψε ΜΟΝΟ τη δική σου περίληψη (πιθανή απάτη ή όχι). "
        "ΜΗΝ αντιγράφεις ετικέτες [BEGIN]/[MIDDLE]/[END], μην αντιγράφεις γραμμές που ξεκινούν με ---, "
        "μην επαναλαμβάνεις το ίδιο το κείμενο της συνομιλίας.\n"
        "Το vishing_probability και το meaning πρέπει να συμφωνούν (αν λες ότι είναι απάτη, ο αριθμός δεν μπορεί να είναι ~0).\n"
        "Πριν το score: σύγκρινε με τα προφίλ 1–4 και μέτρησε τις ισχυρές ενδείξεις (άμεσες ή έμμεσες). "
        "Αν ταιριάζει έστω ένα προφίλ ισχυρά ή συντρέχουν 2+ ισχυρές ενδείξεις, "
        "βάλε vishing_probability τουλάχιστον ~0.55 εκτός αν είναι ξεκάθαρα νόμιμη κλήση χωρίς μυστικά.\n"
        "ΚΡΙΣΙΜΟ: ΜΗΝ γράφεις «Παράδειγμα», «Παράδειγμα απόδειξης», «JSON:» ή αντιγράφεις αριθμούς από τις οδηγίες. "
        "Ξεκίνα απευθείας με το σύμβολο { . Η απάντηση είναι ΜΟΝΟ ένα JSON βασισμένο στη Συνομιλία.\n"
        "Μορφή κλειδιών (όχι copy-paste τιμών — βαθμολόγησε ΕΣΥ): "
        '{"domain":"<μία ετικέτα από τη λίστα>","vishing_probability":<0-1>,"fraud_patterns_detected":[],"meaning":"<ελληνικά>"}\n\n'
        "Συνομιλία:\n"
    )

    prefix = intro + domain_block + output_spec
    budget = _max_classifier_prompt_tokens()
    scale = 1.0
    body = _context_for_model(raw_text, use_full_text, scale)
    full = prefix + body + "\n"
    for _ in range(28):
        if _count_prompt_tokens(full) <= budget:
            return full
        scale *= 0.84
        body = _context_for_model(raw_text, use_full_text, scale)
        full = prefix + body + "\n"

    # Last resort: trim conversation tail (keep instructions intact)
    while _count_prompt_tokens(full) > budget and len(body) > 200:
        body = body[: int(len(body) * 0.88)]
        full = prefix + body + "\n"
    return full


def clamp_01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def apply_accumulated_prob(previous_score: Optional[float], new_llm_prob: float) -> float:
    """Backward-compatible name: progressive blend (may decrease). No segment heuristics."""
    return finalize_progressive_score(
        previous_score, new_llm_prob, latest_segment_text="", full_text=""
    )[0]


def _segment_heuristic_bump(h_seg: float) -> float:
    raw = h_seg * config.SEGMENT_HEURISTIC_SCALE
    return min(config.SEGMENT_HEURISTIC_CAP, raw)


def apply_progressive_prob(
    previous_score: Optional[float],
    llm_prob: float,
    latest_segment_text: str,
) -> float:
    """
    Blend previous score with new LLM estimate (so score can fall on benign turns).
    Add a small bump from fraud-like keywords in the latest segment only.
    """
    if not config.ACCUMULATE_VISHING_PROB:
        h_seg = heuristics.compute_heuristic_score(latest_segment_text.strip()) if latest_segment_text else 0.0
        return clamp_01(llm_prob + _segment_heuristic_bump(h_seg))
    baseline = config.STARTING_VISHING_PROBABILITY
    prior = previous_score if previous_score is not None else baseline
    w = config.PROGRESS_LLM_WEIGHT
    blended = (1.0 - w) * prior + w * llm_prob
    h_seg = heuristics.compute_heuristic_score(latest_segment_text.strip()) if latest_segment_text else 0.0
    return clamp_01(blended + _segment_heuristic_bump(h_seg))


def resolve_ambiguous_band(prob: float, h_seg: float, h_full: float) -> float:
    """
    If probability sits around 50%, nudge toward VISHING or SAFE using heuristics,
    otherwise leave unchanged (stays CRITICAL band visually).
    """
    if not (config.NEAR_FIFTY_LOW <= prob <= config.NEAR_FIFTY_HIGH):
        return prob
    if h_seg >= config.AMBIGUOUS_PRESSURE_SEG or h_full >= config.AMBIGUOUS_PRESSURE_SEG * 2.5:
        return clamp_01(min(0.65, prob + config.AMBIGUOUS_NUDGE_UP))
    if h_seg <= config.AMBIGUOUS_CLEAN_SEG and h_full <= config.AMBIGUOUS_CLEAN_FULL:
        return clamp_01(max(0.2, prob - config.AMBIGUOUS_NUDGE_DOWN))
    return prob


def _nudge_from_full_heuristic(prob: float, h_full: float) -> float:
    """Boost score when keyword/heuristic on full call is strong (fixes missed vishing)."""
    if h_full < config.FULL_HEURISTIC_NUDGE_THRESHOLD:
        return prob
    return clamp_01(prob + config.FULL_HEURISTIC_NUDGE_AMOUNT)


def _subtle_recall_lift(prob: float, h_full: float) -> float:
    """
    When full-text heuristic is in a mid band (below main recall floor) but the LLM
    stayed very SAFE, add a small delta so borderline scams land in CRITICAL+.
    """
    d = float(getattr(config, "SUBTLE_RECALL_LIFT_DELTA", 0.0) or 0.0)
    if d <= 0.0:
        return prob
    lo = float(getattr(config, "SUBTLE_RECALL_LIFT_MIN_H", 0.99))
    hi = float(getattr(config, "SUBTLE_RECALL_LIFT_MAX_H", 0.0))
    mx = float(getattr(config, "SUBTLE_RECALL_LIFT_MAX_PROB", 0.0))
    if lo >= hi or h_full < lo or h_full >= hi or prob > mx:
        return prob
    return clamp_01(prob + d)


def finalize_progressive_score(
    previous_score: Optional[float],
    llm_prob: float,
    latest_segment_text: str,
    full_text: str,
) -> Tuple[float, dict]:
    """
    Full pipeline: progressive blend + ambiguous-band resolution + full-text heuristic nudge.
    Returns (final_prob, debug dict with heuristics).
    """
    lt = (latest_segment_text or "").strip()
    ft = (full_text or "").strip()
    h_seg = heuristics.compute_heuristic_score(lt) if lt else 0.0
    h_full = heuristics.compute_heuristic_score(ft) if ft else h_seg
    p = apply_progressive_prob(previous_score, llm_prob, lt)
    p = resolve_ambiguous_band(p, h_seg, h_full)
    p = _nudge_from_full_heuristic(p, h_full)
    p = _subtle_recall_lift(p, h_full)
    p = apply_low_signal_short_call_cap(p, ft, h_full)
    meta = {"heuristic_segment": round(h_seg, 4), "heuristic_full": round(h_full, 4)}
    return clamp_01(p), meta


def normalize_domain(domain: str) -> str:
    """Map model output to exact allowed domain label."""
    if not domain or not isinstance(domain, str):
        return "N/A"
    d = domain.strip().strip('"').strip("'")
    for allowed in ALLOWED_DOMAINS:
        if d.lower() == allowed.lower():
            return allowed
    d_lower = d.lower()
    # Exact-ish substring match against allowed labels
    for allowed in ALLOWED_DOMAINS:
        al = allowed.lower()
        if al in d_lower or d_lower in al:
            return allowed

    # Greek + semantic hints (model often answers in Greek)
    if any(
        x in d_lower
        for x in (
            "τράπεζ",
            "τραπεζ",
            "banking",
            "λογαριασμ",
            "κάρτα",
            "χρεωστ",
            "πιστωτ",
            "iban",
            "δάνει",
        )
    ):
        return "Banking"
    if any(
        x in d_lower
        for x in (
            "επαλήθευση",
            "επιβεβαίωση ταυτ",
            "verification",
            "κωδικός μιας",
            "μητρικ",
            "πατρικό όνομα",
        )
    ):
        return "Account Verification"
    if any(x in d_lower for x in ("δεη", "ρεύμα", "ρευμα", "νερό", "νερο", "οτε", "cosmote", "wind", "utility", "λογαριασμός νερού")):
        return "Utilities"
    if any(x in d_lower for x in ("ιατρ", "γιατρ", "νοσοκ", "φαρμάκ", "υγεία", "health")):
        return "Healthcare"
    if any(x in d_lower for x in ("σχολ", "μαθητ", "δάσκαλ", "πανεπιστήμι", "education")):
        return "Education"
    if any(x in d_lower for x in ("εκφοβισμ", "bully")):
        return "Bullying"
    if any(x in d_lower for x in ("σεξουαλ", "παρενόχλ", "harass")):
        return "Sexual Harassment"
    if any(x in d_lower for x in ("μίσος", "hate speech", "hate")):
        return "Hate Speech"
    if any(x in d_lower for x in ("neutral", "ουδέτερ", "φυσιολογ", "κανονικ", "φιλικ")):
        return "Neutral"
    return "N/A"


def infer_domain_from_transcript(text: str) -> str:
    """
    When the model omits or garbles domain, infer from conversation keywords (Greek + English).
    """
    if not text or not isinstance(text, str):
        return "N/A"
    tl = text.lower()
    if any(
        x in tl
        for x in (
            "τράπεζ",
            "τραπεζ",
            "banking",
            "λογαριασμ",
            "κάρτα",
            "χρεωστ",
            "πιστωτ",
            "iban",
            "δάνει",
            "pin ",
            " otp",
            "κωδικό",
        )
    ):
        return "Banking"
    if any(
        x in tl
        for x in (
            "επαλήθευση",
            "επιβεβαίωση ταυτ",
            "verification",
            "κωδικός μιας",
            "μητρικ",
            "πατρικό όνομα",
            "ταυτότητα",
        )
    ):
        return "Account Verification"
    if any(x in tl for x in ("δεη", "ρεύμα", "ρευμα", "νερό", "νερο", "οτε", "cosmote", "wind", "utility", "λογαριασμός νερού")):
        return "Utilities"
    if any(x in tl for x in ("ιατρ", "γιατρ", "νοσοκ", "φαρμάκ", "υγεία", "health", "ραντεβού γιατρ")):
        return "Healthcare"
    if any(x in tl for x in ("σχολ", "μαθητ", "δάσκαλ", "πανεπιστήμι", "education")):
        return "Education"
    if any(x in tl for x in ("εκφοβισμ", "bully")):
        return "Bullying"
    if any(x in tl for x in ("σεξουαλ", "παρενόχλ", "harass")):
        return "Sexual Harassment"
    if any(x in tl for x in ("μίσος", "hate speech")):
        return "Hate Speech"
    if any(x in tl for x in ("παραγγελία", "delivery", "ούτε απάτη", "φιλική κλήση")):
        return "Neutral"
    return "N/A"


def finalize_domain(llm_domain: str, full_transcript: str) -> str:
    """Prefer normalized model domain; fall back to transcript heuristics."""
    d = normalize_domain(llm_domain)
    if d != "N/A":
        return d
    return infer_domain_from_transcript(full_transcript)


def _strip_completion_fences(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _iter_json_dicts(raw: str) -> List[Dict[str, Any]]:
    """Scan completion for all top-level JSON objects (handles echoed example + real answer)."""
    s = _strip_completion_fences(raw)
    out: List[Dict[str, Any]] = []
    dec = json.JSONDecoder()
    i = 0
    n = len(s)
    while i < n:
        start = s.find("{", i)
        if start < 0:
            break
        try:
            obj, end = dec.raw_decode(s, start)
            if isinstance(obj, dict):
                out.append(obj)
            i = end
        except (json.JSONDecodeError, ValueError):
            i = start + 1
    return out


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Parse JSON object(s) from model output; prefer the last dict (reduces FP from echoed examples)."""
    dicts = _iter_json_dicts(raw)
    if not dicts:
        return None
    # Prefer last object that looks like a behavior-classifier payload.
    for d in reversed(dicts):
        if any(
            k in d
            for k in ("vishing_probability", "fraud_patterns_detected", "meaning", "analysis", "resolution")
        ):
            return d
    return dicts[-1]


# Prompt example line (model sometimes copies meaning text → treat as echo).
_BEHAVIOR_PROMPT_EXAMPLE_MEANING = "Αίτημα κωδικού με πίεση."


def _completion_smells_like_instruction_echo(completion: str, meaning: str) -> bool:
    """True if output likely repeats the prompt's example JSON / labels (common FP source)."""
    t = (completion or "").strip()
    head = t[:400]
    if re.search(r"Παράδειγμα(\s+απόδειξης)?", head, re.I):
        return True
    if re.match(r"JSON:\s*\{", t, re.I):
        return True
    # Model prefixes JSON with "Απάντηση:" — often pasted template, not a calibrated score.
    if re.search(r"απάντηση\s*:", head, re.I):
        return True
    m = (meaning or "").strip()
    if m and m == _BEHAVIOR_PROMPT_EXAMPLE_MEANING:
        return True
    if m and re.match(r"^απάντηση\s*:", m, re.I):
        return True
    return False


def build_critical_triage_prompt(transcript: str, prior_probability: float) -> str:
    """
    Short second pass for calls in the CRITICAL band: force VISHING vs SAFE vs remain uncertain.
    Uses full head/middle/tail packing like the main classifier.
    """
    raw = (transcript or "").strip()
    prior = clamp_01(float(prior_probability))
    spec = (
        "Είσαι ελεγκτής τηλεφωνικής απάτης (vishing). Η πρώτη ανάλυση έδωσε "
        f"vishing_probability≈{prior:.2f} (μπορεί να είναι λάθος αν πρόκειται για προσποίηση "
        "ΔΕΗ/παρόχου/τράπεζας/σχολείου με αίτημα στοιχείων ή «επαλήθευσης»). "
        "Διάβασε ΟΛΗ τη συνομιλία και αποφάσισε τελικά:\n"
        "- VISHING: σαφές ή πολύ πιθανό vishing (προσποίηση + στόχος/πίεση/απόσπαση ή ισχυρά μοτίβα απάτης).\n"
        "- SAFE: καθαρά καθημερινή ή νόμιμη κλήση χωρίς δόλο· όχι προσποίηση φορέα με αίτημα ευαίσθητων στοιχείων.\n"
        "- CRITICAL_UNCERTAIN: παραμένει αντικειμενικά ασαφές μετά την πλήρη ανάγνωση.\n\n"
        "Μην είσαι υπερβολικά επιφυλακτικός αν υπάρχουν 2+ ισχυρές ενδείξεις· τότε προτίμησε VISHING. "
        "Μην βαφτίζεις VISHING αθώες καθημερινές κουβέντες μόνο επειδή ακούστηκε «τράπεζα» ή «λογαριασμός».\n\n"
        "Απάντηση ΜΟΝΟ με ΈΝΑ JSON (χωρίς markdown): "
        '{"resolution":"VISHING"|"SAFE"|"CRITICAL_UNCERTAIN","meaning":"μία σύντομη πρόταση στα ελληνικά"}\n\n'
        "Συνομιλία:\n"
    )
    triage_completion = int(getattr(config, "VLLM_TRIAGE_COMPLETION_TOKENS", 48))
    budget = _max_prompt_tokens_for_completion(triage_completion)
    scale = 0.88
    body = _context_for_model(raw, use_full_text=True, scale=scale)
    full = spec + body + "\n"
    for _ in range(28):
        if _count_prompt_tokens(full) <= budget:
            return full
        scale *= 0.84
        body = _context_for_model(raw, use_full_text=True, scale=scale)
        full = spec + body + "\n"
    while _count_prompt_tokens(full) > budget and len(body) > 200:
        body = body[: int(len(body) * 0.88)]
        full = spec + body + "\n"
    return full


def parse_critical_triage_completion(raw: str) -> Tuple[str, str]:
    """Returns (resolution in {VISHING, SAFE, CRITICAL_UNCERTAIN}, meaning)."""
    r0 = (raw or "").strip()
    parsed = _extract_json_object(r0) or {}
    res = str(parsed.get("resolution", "")).strip().upper().replace(" ", "_")
    meaning = _sanitize_meaning_text(str(parsed.get("meaning", "")).strip())
    if "VISHING" in res and ("NON" in res or "NOT" in res):
        return "SAFE", meaning
    if "SAFE" in res and "UNSAFE" not in res:
        return "SAFE", meaning
    if "VISHING" in res:
        return "VISHING", meaning
    # Garbage completions (e.g. "Απάντηση: {...}") default to UNCERTAIN → false positives when
    # CRITICAL counts as fraud; treat as non-committal → SAFE unless explicitly VISHING above.
    if re.search(r"απάντηση\s*:", r0[:500], re.I) and "VISHING" not in res:
        return "SAFE", meaning
    return "CRITICAL_UNCERTAIN", meaning


def score_from_critical_triage_resolution(resolution: str) -> float:
    """Map triage label to a calibrated probability (VISHING high, SAFE low, UNCERTAIN ≥ FINAL threshold by default)."""
    r = (resolution or "").strip().upper()
    if r == "VISHING":
        return clamp_01(float(config.CRITICAL_TRIAGE_SCORE_VISHING))
    if r == "SAFE":
        return clamp_01(float(config.CRITICAL_TRIAGE_SCORE_SAFE))
    return clamp_01(float(config.CRITICAL_TRIAGE_SCORE_UNCERTAIN))


def _sanitize_meaning_text(s: str) -> str:
    """Strip transcript section headers the model sometimes copies into `meaning`."""
    if not s:
        return s
    out_lines: List[str] = []
    for ln in s.split("\n"):
        t = ln.strip()
        if not t:
            continue
        if t.startswith("---") or t.startswith("[BEGIN]") or t.startswith("[END]") or t.startswith("[MIDDLE_"):
            continue
        if t.startswith("[OMITTED_MIDDLE"):
            continue
        if re.match(r"^απάντηση\s*:", t, re.I):
            continue
        if re.search(r"απόσπασμα|λέξεις\s+\d+\s*[-–]", t, re.I):
            continue
        out_lines.append(ln)
    out = " ".join(x.strip() for x in out_lines).strip()
    return out if out else s[:220].strip()


def _coerce_prob_if_meaning_asserts_vishing(prob: float, meaning: str) -> float:
    """If the model writes that it is vishing but leaves probability near zero, fix inconsistency."""
    if prob >= 0.4:
        return prob
    ml = (meaning or "").lower()
    if not ml:
        return prob
    if re.search(r"δεν\s+είναι|όχι\s+απάτη|not\s+vishing|όχι\s+vishing", ml):
        return prob
    if re.search(
        r"\bvishing\b|phishing|τηλεφωνικ[ήή]\s+απάτη|είναι\s+απάτη|κλήση\s+απάτης|"
        r"απάτη\b|απατεώνα|απατεων",
        ml,
        re.I,
    ):
        return max(prob, max(0.52, config.FINAL_VISHING_STOP_THRESHOLD))
    return prob


def parse_behavior_completion(completion: str) -> Tuple[List[str], str, float, str]:
    """
    Parse model completion: fraud_patterns_detected, analysis, vishing_probability, domain.
    Returns (patterns, analysis, vishing_prob in [0,1], domain normalized to allowed).
    """
    raw = completion.strip()
    patterns: List[str] = []
    analysis = ""
    vishing_prob = 0.0
    domain = "N/A"

    parsed = _extract_json_object(raw)
    got_prob = False
    if parsed:
        if isinstance(parsed.get("fraud_patterns_detected"), list):
            patterns = [str(p).strip() for p in parsed["fraud_patterns_detected"]]
        if parsed.get("meaning"):
            analysis = _sanitize_meaning_text(str(parsed["meaning"]).strip())
        elif parsed.get("analysis"):
            analysis = _sanitize_meaning_text(str(parsed["analysis"]).strip())
        if parsed.get("vishing_probability") is not None:
            try:
                vishing_prob = clamp_01(float(parsed["vishing_probability"]))
                got_prob = True
            except (TypeError, ValueError):
                pass
        if parsed.get("domain") is not None:
            domain = normalize_domain(str(parsed["domain"]))

    if not got_prob:
        m = re.search(r'"vishing_probability"\s*:\s*([0-9.]+)', raw, re.IGNORECASE)
        if m:
            try:
                vishing_prob = clamp_01(float(m.group(1)))
                got_prob = True
            except (ValueError, TypeError):
                pass
        # Never grab the first float in the whole completion (e.g. inside `meaning`) when JSON exists.
        if not got_prob and parsed is None:
            m2 = re.search(r"\b(0?\.\d+|1(?:\.0+)?)\b", raw)
            if m2:
                try:
                    vishing_prob = clamp_01(float(m2.group(1)))
                except (ValueError, TypeError):
                    pass

    if domain == "N/A":
        dm = re.search(r'"domain"\s*:\s*"([^"]+)"', raw, re.IGNORECASE)
        if dm:
            domain = normalize_domain(dm.group(1))
        if domain == "N/A":
            dm2 = re.search(r"'domain'\s*:\s*'([^']+)'", raw, re.IGNORECASE)
            if dm2:
                domain = normalize_domain(dm2.group(1))

    if not analysis:
        analysis = _sanitize_meaning_text(raw[:200])

    echo = _completion_smells_like_instruction_echo(raw, analysis)
    if echo:
        cap = float(getattr(config, "COMPLETION_ECHO_MAX_PARSED_PROB", 0.24))
        vishing_prob = min(vishing_prob, cap)
    else:
        vishing_prob = _coerce_prob_if_meaning_asserts_vishing(vishing_prob, analysis)

    return patterns, analysis, vishing_prob, domain


def prob_to_risk_status(prob: float) -> str:
    """Map vishing_probability in [0,1] to SAFE / CRITICAL / VISHING."""
    if prob >= config.FINAL_VISHING_STOP_THRESHOLD:
        return "VISHING"
    if prob <= config.SAFE_SCORE_MAX:
        return "SAFE"
    return "CRITICAL"


def is_vishing_label(prob: float) -> bool:
    """True if probability is at or above the final vishing threshold."""
    return prob >= config.FINAL_VISHING_STOP_THRESHOLD


def should_stop_reclassification(prob: float) -> bool:
    """Stop classifying a call once it reaches the final vishing threshold."""
    return prob >= config.FINAL_VISHING_STOP_THRESHOLD


def category_from_risk_status(risk_status: str) -> str:
    """Map risk status to UI category bucket."""
    if risk_status == "VISHING":
        return "vishing"
    if risk_status == "CRITICAL":
        return "critical"
    return "normal"


# --- Behavioral fraud explainability (Prompt #4 decomposed: one LLM call per sub-question) ---
# Does not affect the production classifier path; used only for the optional explainability run.
EXPLAINABILITY_STEPS: List[Dict[str, str]] = [
    {
        "key": "pressure_tactics",
        "title_el": "Υποερώτηση 1 — Πίεση / τακτικές πίεσης",
        "focus": (
            "Αντικείμενο ΜΟΝΟ αυτού του βήματος: κοινωνική πίεση για άμεση ενέργεια χωρίς περιθώριο ελέγχου.\n"
            "Τυπικά σήματα vishing: τεχνητό επείγον, ψυχρός εκβιασμός («τώρα αλλιώς…»), διπλωματική απειλή κλειδώματος λογαριασμού-κάρτας-φορολογίας, "
            "ψευδοπροθεσμίες, απόρριψη του «θα το ελέγξω και επανέρχομαι».\n"
            "Διαχώρισε σκόπιμα: καθημερινή φόρτιση ή πραγματική αναμονή σε ουρά υπηρεσίας ΔΕΝ ισοδυναμεί μόνη της με απάτη — "
            "σημείωσέ το αν το στρές φαίνεται νόμιμο· αν όμως συνδυάζεται με αίτημα μυστικών ή «μην κλείσε», το σήμα βαραίνει.\n"
            "Απάντηση: 2–4 προτάσεις· βαθμός ένδειξης (ασθενής / μέτρια / ισχυρή / δεν διακρίνεται) και σύντομο γιατί, χωρίς μεταγραφή λόγων."
        ),
    },
    {
        "key": "authority_impersonation",
        "title_el": "Υποερώτηση 2 — Προσποίηση εξουσίας / φορέα",
        "focus": (
            "Αντικείμενο ΜΟΝΟ: ισχυρισμός ρόλου/φορέα (τράπεζα, δημόσια υπηρεσία, ενέργεια, courier, νοσοκομείο, «ασφάλεια λογαριασμού», "
            "«τεχνική υποστήριξη», «εισαγγελέας», «μεγάλη πλατφόρμα») χωρίς αποδεικνύσιμη επαλήθευση.\n"
            "Σήματα: κρύα κλήση που «ξέρει» λεπτομέρειες, άρνηση να δώσει στοιχεία επικοινωνίας προς έλεγχο, οδηγία «μην πεις σε άλλον», "
            "αλλαγή αναφερόμενου φορέα μέσα στην ίδια ροή.\n"
            "Απάντηση: 2–4 προτάσεις· το κατά πόσο η παρουσία φορέα φαίνεται γνήσια ή ύποπτη, με βαθμό ένδειξης· χωρίς ονόματα ή αποσπάσματα."
        ),
    },
    {
        "key": "information_extraction",
        "title_el": "Υποερώτηση 3 — Απόσπαση / συλλογή πληροφοριών",
        "focus": (
            "Αντικείμενο ΜΟΝΟ: προσπάθεια συλλογής ευαίσθητων δεδομένων ή ελέγχου συστήματος μέσω φωνής.\n"
            "Σήματα: κωδικοί, PIN, OTP, «μυστικός κωδικός τραπέζης», πλήρη στοιχεία κάρτας, κωδικός e-banking, απάντηση σε «δοκιμαστικά» ερωτήματα "
            "ασφαλείας, ανάγνωση κώδικα από μήνυμα, εγκατάσταση εφαρμογής ή άνοιγμα συνδέσμου «για επαλήθευση», απομακρυσμένη πρόσβαση, "
            "αλλαγή στοιχείων επικοινωνίας λογαριασμού στην κλήση.\n"
            "Νόμιμη εξυπηρέτηση σπάνια ζητά πλήρη μυστικά στο τηλέφωνο· αν ζητείται, σημείωσε αν επιτρέπεται έλεγχος από γνωστό νούμερο-κατάστημα.\n"
            "Απάντηση: 2–4 προτάσεις· βαθμός ένδειξης απόσπασης· χωρίς επανάληψη αιτημάτων αυτολεξεί."
        ),
    },
    {
        "key": "trust_building_manipulation",
        "title_el": "Υποερώτηση 4 — Χειραγώγηση εμπιστοσύνης",
        "focus": (
            "Αντικείμενο ΜΟΝΟ: τεχνική χαλάρωσης άμυνας πριν από ρίσκο-αιτήματα.\n"
            "Σήματα: υπερβολικά οικεία ύφος, «είμαστε δίπλα σου», δήθεν εμπιστευτικότητα, επίσπευση εμπιστοσύνης, "
            "μιμείται διαδικασίες νομιμότητας (αριθμοί πρωτοκόλλου, ψευδο-πρωτόκολλα) χωρίς να αντέχουν σε έλεγχο, "
            "απόσπαση προσοχής από κόκκινες σημαίες (π.χ. προσωπικές λεπτομέρειες που μειώνουν σκεπτικισμό).\n"
            "Απάντηση: 2–4 προτάσεις· βαθμός ένδειξης χειραγώγησης· χωρίς ψυχογραφική αφήγηση προσώπων της συνομιλίας."
        ),
    },
    {
        "key": "unusual_flow",
        "title_el": "Υποερώτηση 5 — Ασυνήθης / ύποπτη ροή συνομιλίας",
        "focus": (
            "Αντικείμενο ΜΟΝΟ: μακροδομή συνομιλίας — ευθυγράμμιση με νόμιμη ροή εξυπηρέτησης έναντι σεναρίου απάτης.\n"
            "Σήματα: άρνηση ή αποφυγή επανάκλησης από γνωστό κανάλι, μεταπήδηση μεταξύ άσχετων απειλών-αιτημάτων, "
            "επανάληψη του ίδιου αιτήματος μετά από αμυντική, αντίφαση σε προηγούμενες δηλώσεις, «σενάριο σενάριου» (πρώτα φόβος μετά η «λύση»), "
            "παράλογη εμμονή σε μία ενέργεια που δεν ταιριάζει στο δηλωμένο ρόλο.\n"
            "Απάντηση: 2–4 προτάσεις· αν η ροή ενισχύει, αποδυναμώνει ή αφήνει ουδέτερη την υποψία vishing, με βαθμό ένδειξης· χωρίς χρονολόγιο αναφοράς."
        ),
    },
    {
        "key": "final_synthesis",
        "title_el": "Υποερώτηση 6 — Σύνθεση (τελική κρίση)",
        "focus": "",
    },
]


_EXPLAIN_OUTPUT_BASE = (
    "Η απάντησή σου περιέχει ΑΠΟΚΛΕΙΣΤΙΚΑ δική σου ανάλυση· μηδεμία μεταγραφή από τη συνομιλία (ο χρήστης την έχει ήδη).\n"
    "ΜΟΝΟ ελληνικά — καμία αγγλική λέξη ή πρόταση.\n"
    "Όχι JSON, όχι αγκύλες {}, όχι markdown. Όχι αρίθμηση (1. 2.), όχι κουκκίδες, όχι λίστες.\n"
    "Μην χρησιμοποιείς ονόματα προσώπων ούτε δεικτική απεύθυνση («Μαρία», «εσύ» σε ρόλο πελάτη). "
    "Μην περιγράφεις συναισθηματικά ή χιουμοριστικά («αστείο», «το πιο…»). Τόνος ψυχρού αναλυτή ασφάλειας.\n"
    "Μην επαναλαμβάνεις οδηγίες λόγου («σύνολο 400 λέξεων», «συνόλο της απάντησής σου» κ.λπ.).\n"
    "ΑΠΑΓΟΡΕΥΕΤΑΙ να αντιγράφεις, να μεταφράζεις κοντά στο πρωτότυπο ή να επαναλαμβάνεις φράσεις που εμφανίζονται "
    "στο κείμενο αναφοράς· ούτε διάλογος σε εισαγωγικά, ούτε «ο ένας λέει… ο άλλος λέει…».\n"
    "Μόνο κριτική γνώμη με τόνους: «Φαίνεται…», «Δεν φαίνεται…», «Υπάρχει πίεση για…», «Δεν διακρίνεται…», "
    "«Η συνομιλία δείχνει…» (χωρίς να αναφέρεις τι ακριβώς ειπώθηκε).\n"
    "ΑΠΑΓΟΡΕΥΕΤΑΙ caller:/agent: και αποσπάσματα κειμένου από τη συνομιλία.\n"
    "Ξεκίνα απευθείας με την ουσία — χωρίς τίτλο, προοίμιο ή μετα-σχόλια.\n"
)

# Placed immediately AFTER the transcript in the prompt so the model sees it last (reduces “continuation” of dialogue).
_EXPLAIN_SUFFIX_SUB = (
    "\n———\n"
    "ΑΠΟΤΕΛΕΣΜΑ ΕΞΟΔΟΥ (υποχρεωτικό):\n"
    "Η απάντησή σου ΔΕΝ είναι συνέχεια του παραπάνω κειμένου. Μην γράφεις νέο διάλογο, μην συνεχίζεις ρόλους, "
    "μην βάζεις caller:/agent: ή ομιλητές.\n"
    "Απάντησε μόνο με 2–4 προτάσεις αφηρημένης κρίσης («φαίνεται / δεν διακρίνεται» + βαθμός ένδειξης). "
    "Ξεκίνα αμέσως με την πρώτη πρόταση της κρίσης σου — όχι επανάληψη αυτού του μνημονίου.\n"
)

# Keep this block short and avoid words the model tends to echo as “the answer” (e.g. «ΑΠΟΤΕΛΕΣΜΑ»).
_EXPLAIN_SUFFIX_SUB_YESNO = (
    "\n<<<\n"
    "Write nothing before or after. Output must be exactly one Greek word: ΝΑΙ or ΟΧΙ. "
    "Never YES/NO, never labels, never English, no punctuation.\n"
)

_EXPLAIN_OUTPUT_YESNO_SUB = (
    "Η τελική απάντηση είναι μόνο μία λέξη στα ελληνικά: ΝΑΙ ή ΟΧΙ (κεφαλαία). "
    "Όχι YES/NO, όχι πρόταση, όχι εξήγηση. Αν δεν είναι σαφές, διάλεξε ΟΧΙ.\n\n"
)

_EXPLAIN_SUFFIX_FINAL = (
    "\n———\n"
    "ΑΠΟΤΕΛΕΣΜΑ ΕΞΟΔΟΥ (υποχρεωτικό):\n"
    "Η απάντησή σου ΔΕΝ αντιγράφει το κείμενο πάνω. Σύνθεση μόνο από τα μοτίβα — χωρίς μεταγραφή ρεπλίκων. "
    "Ξεκίνα αμέσως με την πρώτη πρόταση της σύνθεσής σου.\n"
)

_EXPLAIN_LENGTH_SUB = "Μήκος: 2 έως 4 σύντομες προτάσεις (περίπου 45–100 λέξεις συνολικά). Όχι μεγαλύτερη έκταση.\n"

_EXPLAIN_LENGTH_FINAL = (
    "Μήκος τελικής σύνθεσης: 6 έως 10 σύντομες προτάσεις (περίπου 160–260 λέξεις), μία ενιαία παράγραφος ή δύο. "
    "Όχι λίστες. Μετά τελειώνεις με μία μόνο πρόταση «Εκτίμηση πιθανότητας vishing: 0,XX».\n"
)


def _classifier_alignment_block(classifier_hint: Optional[Dict[str, Any]]) -> str:
    """Inject main classifier outcome so synthesis cannot contradict VISHING/CRITICAL vs SAFE."""
    if not classifier_hint or not isinstance(classifier_hint, dict):
        return ""
    rs = str(classifier_hint.get("risk_status") or "").strip()
    if rs == "ERROR":
        return (
            "=== Κύριος ταξινομητής ===\n"
            "Αποτέλεσμα: ΣΦΑΛΜΑ — δώσε τελική κρίση μόνο από τις πέντε υποερωτήσεις, χωρίς να ισχυρίζεσαι "
            "ότι έχεις το επίσημο νούμερο του ταξινομητή.\n\n"
        )
    score_v: Optional[float] = None
    try:
        if classifier_hint.get("score") is not None:
            score_v = float(classifier_hint["score"])
    except (TypeError, ValueError):
        score_v = None
    sc_s = f"{score_v:.3f}" if score_v is not None else "άγνωστο"
    lbl = str(classifier_hint.get("label") or "—").strip()
    dom = str(classifier_hint.get("domain") or "—").strip()
    final_t = float(getattr(config, "FINAL_VISHING_STOP_THRESHOLD", 0.52))
    safe_t = float(getattr(config, "SAFE_SCORE_MAX", 0.28))
    return (
        "=== Αποτέλεσμα κύριου ταξινομητή (υποχρεωτική συνέπεια) ===\n"
        f"Βαθμός vishing (0–1): {sc_s}\n"
        f"Κατάσταση κινδύνου: {rs or '—'}\n"
        f"Ετικέτα σήματος απάτης: {lbl}\n"
        f"Τομέας που έβγαλε ο ταξινομητής: {dom}\n"
        "\n"
        "ΚΑΝΟΝΕΣ (όχι αντιφάσεις):\n"
        f"- Αν η κατάσταση είναι VISHING ή το score ≥ {final_t:.2f}, απαγορεύεται να γράφεις ότι η κλήση είναι ασφαλής/αθώα "
        "και η τελική «Εκτίμηση πιθανότητας vishing» πρέπει να ταιριάζει (τουλάχιστον κοντά) στο βαθμό του ταξινομητή — "
        "όχι 0,00–0,15.\n"
        f"- Αν η κατάσταση είναι CRITICAL (ενδιάμεσο, περίπου μεταξύ {safe_t:.2f} και {final_t:.2f}), περιέγραψε αμφισημία και "
        "κίνδυνο· όχι «απόλυτα καθαρή κλήση».\n"
        f"- Αν η κατάσταση είναι SAFE και score ≤ {safe_t:.2f}, μόνο τότε μπορείς να μιλάς για χαμηλό/μηδαμινό κίνδυνο και "
        "τελική εκτίμηση κοντά σε αυτό το βαθμό.\n"
        "- Η σύνθεσή σου εξηγεί γιατί ο ταξινομητής έβγαλε αυτό το αποτέλεσμα, όχι το αντίθετο.\n\n"
    )

# Lines shorter/longer than this ratio (after one contains the other) count as “echo” of transcript.
_EXPLAIN_TRANSCRIPT_CONTAINMENT_MIN = 0.88


def _squeeze_ws_lower(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _drop_verbatim_transcript_lines_only(text_block: str, transcript: str) -> str:
    """Remove only lines that exactly match a transcript line (after whitespace collapse). Less aggressive than fuzzy overlap."""
    tr = (transcript or "").strip()
    tb = (text_block or "").strip()
    if not tr or not tb or len(tr) <= 40:
        return tb
    tlines = [ln.strip() for ln in tr.splitlines() if len(ln.strip()) >= 22]
    if not tlines:
        return tb
    out_lines: List[str] = []
    for line in tb.splitlines():
        ls = line.strip()
        if not ls:
            out_lines.append(line)
            continue
        drop = False
        sls = _squeeze_ws_lower(ls)
        for tl in tlines:
            if sls == _squeeze_ws_lower(tl):
                drop = True
                break
        if not drop:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


def sanitize_explainability_completion(completion: str, transcript: str = "") -> str:
    """
    Remove role-prefixed dialogue lines and lines/sentences that largely match the transcript
    so the dashboard shows only free-standing model analysis.
    """
    text = (completion or "").strip()
    if not text:
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Inline / glued turns: "…text. agent: … caller: …" — strip every agent:/caller: segment to next marker or EOS.
    _ROLE_MARK = re.compile(
        r"(?iu)(caller|agent|καλών|πράκτορας|speaker|user|assistant)\b\s*[:：]\s*",
    )

    def _strip_role_marked_chunks(s: str) -> str:
        """Keep text outside agent:/caller: … segments (each segment runs to the next tag or end)."""
        out_parts: List[str] = []
        i = 0
        while i < len(s):
            m = _ROLE_MARK.search(s, i)
            if not m:
                out_parts.append(s[i:])
                break
            out_parts.append(s[i : m.start()])
            n = _ROLE_MARK.search(s, m.end())
            if n:
                i = n.start()
            else:
                break
        merged = "".join(out_parts)
        return re.sub(r"[ \t]+\n", "\n", merged).strip()

    text = _strip_role_marked_chunks(text)

    role_line = re.compile(
        r"^\s*(caller|agent|καλών|πράκτορας|speaker|user|assistant|χρήστης)\s*[:：]\s*",
        re.I | re.UNICODE,
    )
    bracket_stage = re.compile(
        r"^\s*\[(BEGIN|END|MIDDLE_\d+|OMITTED_MIDDLE[^\]]*)\]\s*$",
        re.I,
    )

    lines_kept: List[str] = []
    for line in text.splitlines():
        st = line.strip()
        if role_line.match(st):
            continue
        if bracket_stage.match(st):
            continue
        # Same line: "…analysis. agent: quoted dialogue…" — keep only before first role tag.
        m_cut = _ROLE_MARK.search(line)
        if m_cut:
            line = line[: m_cut.start()].rstrip()
        if line:
            lines_kept.append(line.rstrip())

    text = "\n".join(lines_kept)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Second pass: chunk strip (model may reintroduce after line join)
    text = _strip_role_marked_chunks(text)
    after_roles = text

    tr = (transcript or "").strip()
    if tr and len(tr) > 40:
        tlines = [ln.strip() for ln in tr.splitlines() if len(ln.strip()) >= 22]
        if tlines:
            out_lines: List[str] = []
            for line in text.splitlines():
                ls = line.strip()
                if not ls:
                    out_lines.append(line)
                    continue
                drop = False
                sls = _squeeze_ws_lower(ls)
                for tl in tlines:
                    if len(ls) < 28:
                        break
                    stl = _squeeze_ws_lower(tl)
                    if sls == stl:
                        drop = True
                        break
                    if len(ls) >= 40 and len(tl) >= 40:
                        if ls in tl or tl in ls:
                            shorter = min(len(ls), len(tl))
                            longer = max(len(ls), len(tl))
                            if longer > 0 and shorter >= longer * _EXPLAIN_TRANSCRIPT_CONTAINMENT_MIN:
                                drop = True
                                break
                if not drop:
                    out_lines.append(line)
            text = "\n".join(out_lines).strip()

    if not text:
        # Model may have paraphrased analysis; aggressive overlap can over-delete. Retry verbatim-line-only removal.
        light = _drop_verbatim_transcript_lines_only(after_roles, tr)
        if len(light) >= 30:
            text = light

    if not text:
        return "Δεν απέμεινε καθαρή ανάλυση μετά το φίλτρο (το μοντέλο επανέλαβε κυρίως διάλογο). Δοκιμάστε ξανά explainability."
    return text


def custom_explain_sub_style_is_yesno() -> bool:
    st = getattr(config, "CUSTOM_EXPLAIN_SUB_STYLE", "prose") or "prose"
    st = str(st).strip().lower()
    return st in ("yesno", "nai_oxi", "naioxi", "binary", "ναι_οχι")


_YESNO_JUNK_TOKENS = frozenset(
    {
        "αποτέλεσμα",
        "αποτελεσμα",
        "αποτελέσματος",
        "συνόλου",
        "συνολου",
        "εξόδου",
        "εξοδου",
        "υποχρεωτικό",
        "υποχρεωτικο",
        "απάντηση",
        "απαντηση",
        "αποτελέσματα",
        "output",
        "final",
        "answer",
        "conversation",
    }
)


def _yesno_classify_word(token: str) -> Optional[str]:
    """Return 'ΝΑΙ' / 'ΟΧΙ' or None for token (after basic cleanup)."""
    w = (token or "").strip()
    if not w:
        return None
    low = w.lower()
    if low in ("ναι", "ναί", "yes", "y"):
        return "ΝΑΙ"
    if low in ("οχι", "όχι", "οχί", "no", "n"):
        return "ΟΧΙ"
    up = w.upper()
    if up in ("ΝΑΙ", "ΝΑΊ"):
        return "ΝΑΙ"
    if up in ("ΟΧΙ", "ΌΧΙ"):
        return "ΟΧΙ"
    return None


def normalize_custom_sub_yesno_completion(completion: str) -> str:
    """
    Collapse model output to ΝΑΙ or ΟΧΙ. Scans all tokens; ignores instruction-echo junk.
    If unclear, returns ΟΧΙ (matches rubric: when in doubt, NO).
    """
    raw = (completion or "").strip()
    if not raw:
        return "ΟΧΙ"
    if raw.startswith("[σφάλμα") or raw.startswith("Δεν απέμεινε"):
        return raw
    t = raw.translate(str.maketrans("", "", '.,;:!?«»"\'…'))
    for w in t.split():
        low = w.lower()
        if low in _YESNO_JUNK_TOKENS or low.startswith("αποτελ"):
            continue
        hit = _yesno_classify_word(w)
        if hit:
            return hit
    yes_m = re.search(r"(?iu)\b(yes|ναι|ναί|y)\b", raw)
    no_m = re.search(r"(?iu)\b(no|οχι|όχι|οχί)\b", raw)
    if yes_m and no_m:
        return "ΝΑΙ" if yes_m.start() < no_m.start() else "ΟΧΙ"
    if yes_m:
        return "ΝΑΙ"
    if no_m:
        return "ΟΧΙ"
    return "ΟΧΙ"


def _rewrite_focus_yesno_greek(focus: str) -> str:
    """
    User sub-prompts often say YES/NO — align with Greek-only output to reduce mixed-language drift.
    """
    t = focus or ""
    patterns: List[Tuple[str, str]] = [
        (r"(?i)\breply\s+with\s+one\s+word\s+only:\s*yes\s+or\s+no\.?", "Απάντησε μόνο με μία λέξη: ΝΑΙ ή ΟΧΙ"),
        (r"(?i)\bone\s+word\s+only:\s*yes\s+or\s+no\.?", "Μία λέξη μόνο: ΝΑΙ ή ΟΧΙ"),
        (r"(?i)\byes\s+or\s+no\b", "ΝΑΙ ή ΟΧΙ"),
        (r"(?i)\byes\b", "ΝΑΙ"),
        (r"(?i)\bno\b", "ΟΧΙ"),
    ]
    for pat, rep in patterns:
        t = re.sub(pat, rep, t)
    return t


def explainability_step_count() -> int:
    return len(EXPLAINABILITY_STEPS)


def explainability_step_spec(step_index: int) -> Dict[str, str]:
    return EXPLAINABILITY_STEPS[step_index]


def explainability_max_tokens_for_step(step_index: int) -> int:
    n = len(EXPLAINABILITY_STEPS)
    if step_index < 0 or step_index >= n:
        return int(getattr(config, "VLLM_EXPLAINABILITY_STEP_TOKENS", 220))
    if step_index == n - 1:
        return max(96, int(getattr(config, "VLLM_EXPLAINABILITY_FINAL_TOKENS", 380)))
    return max(64, int(getattr(config, "VLLM_EXPLAINABILITY_STEP_TOKENS", 220)))


def _max_explain_prompt_token_budget(step_index: int) -> int:
    mt = explainability_max_tokens_for_step(step_index)
    return _max_prompt_tokens_for_completion(mt)


def build_explainability_prompt(
    text: str,
    step_index: int,
    prior_step_raw: List[str],
    use_full_text: bool = True,
    classifier_hint: Optional[Dict[str, Any]] = None,
    sub_focus_override: Optional[str] = None,
) -> str:
    """
    One focused prompt per sub-question; answers are plain Greek prose (no JSON) for dashboard explainability.
    prior_step_raw: model outputs from steps 0..n-2 (for the synthesis step).
    """
    n = len(EXPLAINABILITY_STEPS)
    if step_index < 0 or step_index >= n:
        raise ValueError("explainability step_index out of range")
    raw_text = (text or "").strip()
    spec = EXPLAINABILITY_STEPS[step_index]
    budget = _max_explain_prompt_token_budget(step_index)
    transcript_header = (
        "Κείμενο συνομιλίας (μόνο για ανάγνωση — ΔΕΝ το επαναλαμβάνεις και ΔΕΝ το παραθέτεις στην απάντησή σου):\n"
    )

    if step_index < n - 1:
        sub_focus = (sub_focus_override or "").strip() or spec["focus"]
        role = (
            "Είσαι αναλυτής τηλεφωνικών συνομιλιών — ανίχνευση απάτης (vishing).\n"
            "Θα διαβάσεις εσωτερικά το κείμενο. Η απάντησή σου ΔΕΝ είναι συνέχεια ή συμπλήρωση του διαλόγου· "
            "δεν αναφέρεις τι ειπώθηκε λέξη-προς-λέξη ούτε γράφεις νέα ρεπλίκα ομιλητή.\n\n"
            + sub_focus
            + "\n\n"
        )
        out_spec = (
            _EXPLAIN_OUTPUT_BASE
            + _EXPLAIN_LENGTH_SUB
            + "Για το συγκεκριμένο μοτίβο: δώσε αιτιολογημένη κρίση με στιλ «Φαίνεται…», «Δεν διακρίνεται…», «Η ροή δείχνει…». "
            "Συμπέρασμα μέσα στο κείμενο: αν αυτό το μοτίβο ανεβάζει, κατεβάζει ή αφήνει ουδέτερη την υποψία vishing — "
            "όχι ξεχωριστός τίτλος, όχι μονολεκτικό Ναι/Όχι μόνο του.\n\n"
        )
        prefix = role + out_spec + transcript_header
    else:
        priors_block = ""
        if prior_step_raw:
            chunks = []
            for idx, pr in enumerate(prior_step_raw):
                chunks.append(f"--- Σύνοψη υποερώτησης {idx + 1} (ήδη ανάλυση LLM, όχι διάλογος) ---\n{(pr or '').strip()}")
            priors_block = (
                "Προηγούμενες αναλύσεις ανά μοτίβο (σύνθεσέ τις· μην τις αντιγράφεις αυτούσιες, μην επαναλάβεις διάλογο):\n"
                + "\n\n".join(chunks)
                + "\n\n"
            )
        align = _classifier_alignment_block(classifier_hint)
        role = (
            "Είσαι αναλυτής τηλεφωνικών συνομιλιών — απάτη/vishing.\n"
            "Σύνθεσε τελική κρίση μόνο στα ελληνικά από τις συνόψεις παραπάνω και την εσωτερική σου αξιολόγηση — "
            "όχι παράθεση αποσπασμάτων, όχι αγγλικά.\n"
            "Αν δόθηκε αποτέλεσμα κύριου ταξινομητή, η τελική εκτίμηση και ο τόνος πρέπει να είναι συνεπή μαζί του "
            "(όχι «ασφαλής κλήση» όταν ο ταξινομητής λέει VISHING).\n\n"
        )
        out_spec = (
            _EXPLAIN_OUTPUT_BASE
            + _EXPLAIN_LENGTH_FINAL
            + "Σύνθεση σε ενιαίο κείμενο: ενοποίησε τάση για πίεση, προσποίηση φορέα, απόσπαση στοιχείων, "
            "χειραγώγηση εμπιστοσύνης, ασυνήθη ροή — χωρίς ξεχωριστές επικεφαλίδες.\n"
            "Κλίμακα: 0,0 φυσιολογική κλήση, 0,5 ύποπτη, 1,0 πολύ πιθανή απάτη.\n"
            "Κλείσε με μία μόνο πρόταση: «Εκτίμηση πιθανότητας vishing: 0,XX» (αριθμός 0–1 με υποδιαστολή). "
            "Το 0,XX πρέπει να πλησιάζει τον βαθμό του ταξινομητή όταν αυτός δόθηκε· όχι εικονικά μηδενικά όταν ο βαθμός είναι υψηλός.\n\n"
        )
        prefix = role + align + priors_block + out_spec + transcript_header

    scale = 1.0
    suffix = _EXPLAIN_SUFFIX_SUB if step_index < n - 1 else _EXPLAIN_SUFFIX_FINAL
    body = _context_for_model(raw_text, use_full_text, scale)
    full = prefix + body + suffix
    for _ in range(28):
        if _count_prompt_tokens(full) <= budget:
            return full
        scale *= 0.84
        body = _context_for_model(raw_text, use_full_text, scale)
        full = prefix + body + suffix
    while _count_prompt_tokens(full) > budget and len(body) > 200:
        body = body[: int(len(body) * 0.88)]
        full = prefix + body + suffix
    return full


def build_custom_explain_chain_turn(
    transcript: str,
    user_focus: str,
    prior_step_raw: List[str],
    *,
    turn_index: int,
) -> str:
    """
    One Prompt-#4-style explainability sub-turn using ``user_focus`` (and optional prior outputs),
    with the same output constraints as built-in explainability substeps.
    """
    raw_text = (transcript or "").strip()
    focus = (user_focus or "").strip()
    if not focus:
        raise ValueError("user_focus required")
    _yesno = custom_explain_sub_style_is_yesno()
    if _yesno:
        focus = _rewrite_focus_yesno_greek(focus)

    priors_block = ""
    if prior_step_raw:
        if _yesno:
            chunks = [
                f"--- Προηγούμενη απάντηση {idx + 1} (μόνο ΝΑΙ ή ΟΧΙ) ---\n{(pr or '').strip()}"
                for idx, pr in enumerate(prior_step_raw)
            ]
            priors_block = (
                "Προηγούμενες απαντήσεις (χωρίς αντιγραφή διαλόγου):\n"
                + "\n\n".join(chunks)
                + "\n\n"
            )
        else:
            chunks = [
                f"--- Προηγούμενο βήμα {idx + 1} (δική σου ανάλυση, όχι διάλογος) ---\n{(pr or '').strip()}"
                for idx, pr in enumerate(prior_step_raw)
            ]
            priors_block = (
                "Προηγούμενα βήματα (σύντομα· χρησιμοποίησέ τα ως πλαίσιο, χωρίς αντιγραφή):\n"
                + "\n\n".join(chunks)
                + "\n\n"
            )

    role = (
        "Είσαι αναλυτής τηλεφωνικών συνομιλιών — ανίχνευση απάτης (vishing).\n"
        "Θα διαβάσεις εσωτερικά το κείμενο. Η απάντησή σου ΔΕΝ είναι συνέχεια ή συμπλήρωση του διαλόγου· "
        "δεν αναφέρεις τι ειπώθηκε λέξη-προς-λέξη ούτε γράφεις νέα ρεπλίκα ομιλητή.\n\n"
        f"Βήμα {turn_index + 1} — εστίαση χρήστη:\n{focus}\n\n"
    )
    transcript_header = (
        "Κείμενο συνομιλίας (μόνο για ανάγνωση — ΔΕΝ το επαναλαμβάνεις και ΔΕΝ το παραθέτεις στην απάντησή σου):\n"
    )
    if _yesno:
        out_spec = _EXPLAIN_OUTPUT_YESNO_SUB
        suffix = _EXPLAIN_SUFFIX_SUB_YESNO
    else:
        out_spec = (
            _EXPLAIN_OUTPUT_BASE
            + _EXPLAIN_LENGTH_SUB
            + "Για αυτή την εστίαση: δώσε αιτιολογημένη κρίση με στιλ «Φαίνεται…», «Δεν διακρίνεται…», «Η ροή δείχνει…». "
            "Συμπέρασμα μέσα στο κείμενο για το πώς επηρεάζει την υποψία vishing — "
            "όχι ξεχωριστός τίτλος, όχι μονολεκτικό Ναι/Όχι μόνο του.\n\n"
        )
        suffix = _EXPLAIN_SUFFIX_SUB
    prefix = priors_block + role + out_spec + transcript_header
    mt_custom_sub = custom_explain_sub_max_tokens()
    budget = _max_prompt_tokens_custom_explain(mt_custom_sub)
    scale = 1.0
    use_full_text = True
    body = _context_for_model(raw_text, use_full_text, scale)
    full = prefix + body + suffix
    for _ in range(28):
        if _count_prompt_tokens(full) <= budget:
            return full
        scale *= 0.84
        body = _context_for_model(raw_text, use_full_text, scale)
        full = prefix + body + suffix
    while _count_prompt_tokens(full) > budget and len(body) > 200:
        body = body[: int(len(body) * 0.88)]
        full = prefix + body + suffix
    return full


# Appended *after* the transcript so the model sees it last (reduces “continue the dialogue”).
_CUSTOM_FULL_TRANSCRIPT_HEADER = (
    "Κείμενο συνομιλίας (ΜΟΝΟ για ανάγνωση — όχι συνέχεια του διαλόγου):\n"
)
# Short, no bullets — models often mimic bullet “rules” and drift into meta‑instruction loops.
_CUSTOM_FULL_OUTPUT_GUARD = (
    "\n<<< END OF CALL TRANSCRIPT >>>\n"
    "Now output ONLY what the task at the very top asks for. "
    "No dialogue, no caller:/agent:, no bullet lists, no restating or quoting these lines or the task wording.\n"
)

_CUSTOM_FULL_OUTPUT_GUARD_JSON = (
    "\n<<< END OF CALL TRANSCRIPT >>>\n"
    "Follow exactly the output format from the instructions at the top (e.g. JSON only). "
    "Do not continue the dialogue. No caller:/agent: lines.\n"
)


def sanitize_custom_full_completion(completion: str) -> str:
    """
    Last pass for «ένα prompt ανά κλήση»: drop meta bullet‑lists («Δεν χρειάζεται… πρότυπο…»)
    and truncate degenerate repetition chains common with small instruct models.
    """
    t = (completion or "").strip()
    if not t:
        return t
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    out_lines: List[str] = []
    for line in t.split("\n"):
        st = line.strip()
        if not st:
            out_lines.append(line)
            continue
        first = st[0]
        if first in "•-*":
            low = st.lower()
            if (
                "δεν χρειάζεται" in low
                or "πρότυπο" in low
                or ("απάντησ" in low and "σου" in low and "περιλαμβάν" in low)
            ):
                continue
        out_lines.append(line.rstrip())
    t = "\n".join(out_lines).strip()
    t = re.sub(r"\n{3,}", "\n\n", t)

    low = t.lower()
    needle = "και να περιλαμβάνεις"
    if needle in low:
        positions = [m.start() for m in re.finditer(re.escape(needle), low)]
        if len(positions) >= 2:
            t = t[: positions[1]].strip()

    low = t.lower()
    if "πρότυπο" in low and low.count("πρότυπο") >= 2:
        cut = t.find("•")
        if cut > 0:
            t = t[:cut].strip()

    t = t.strip()
    if not t:
        return "Δεν απέμεινε χρήσιμο κείμενο μετά το φίλτρο (το μοντέλο έβγαλε μόνο μετα‑οδηγίες). Δοκιμάστε ξανά ή συντομεύστε το prompt."
    return t


def build_custom_explainability_prompt(
    custom_prompt: str,
    transcript: str,
    *,
    output_format: str = "prose",
) -> str:
    """
    User instructions, then transcript, then a fixed output guard (after the dialogue) so the
    model is less likely to continue role-tagged dialogue instead of answering.

    output_format: \"prose\" (default) or \"json\" — lighter tail when the user prompt requests raw JSON.
    """
    cp = (custom_prompt or "").strip()
    raw_text = (transcript or "").strip()
    fmt = (output_format or "prose").strip().lower()
    suffix = _CUSTOM_FULL_OUTPUT_GUARD_JSON if fmt == "json" else _CUSTOM_FULL_OUTPUT_GUARD
    mt_custom_full = custom_explain_main_max_tokens()
    budget = _max_prompt_tokens_custom_explain(mt_custom_full)
    scale = 1.0
    body = _context_for_model(raw_text, True, scale)
    mid = _CUSTOM_FULL_TRANSCRIPT_HEADER
    prefix = cp + "\n\n" + mid
    full = prefix + body + suffix
    for _ in range(28):
        if _count_prompt_tokens(full) <= budget:
            return full
        scale *= 0.84
        body = _context_for_model(raw_text, True, scale)
        full = prefix + body + suffix
    while _count_prompt_tokens(full) > budget and len(body) > 200:
        body = body[: int(len(body) * 0.88)]
        full = prefix + body + suffix
    cp_work = cp
    while _count_prompt_tokens(full) > budget and len(cp_work) > 400:
        cp_work = cp_work[: int(len(cp_work) * 0.9)].rstrip()
        prefix = cp_work + "\n\n" + mid
        full = prefix + body + suffix
    return full
