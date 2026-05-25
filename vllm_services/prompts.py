"""
Prompt builders for vishing detection and call classification.

- build_yes_no_prompt: zero-shot YES/NO vishing classification (from experiment1_mistral)
- build_probability_prompt: JSON probability output (from experiment_utils)
- build_classification_prompt: category/sentiment classification (from classifier prompt.txt)
- PromptLoader: hot-reloadable prompt file (from existing classifier)
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def build_yes_no_prompt(text: str) -> str:
    """
    Build a classification prompt for zero-shot vishing detection.

    The model must answer ONLY with YES (vishing) or NO (not vishing).
    Do not bias toward YES when uncertain (avoid false positives on normal calls).

    Ported from experiment1_mistral.build_prompt().
    """
    instructions = (
        "You are a fraud detection system specialized in detecting Greek vishing "
        "(voice phishing) phone calls.\n\n"
        "You will be given a full call transcript in Greek.\n"
        "Your task is to decide if this call is a vishing (fraud) attempt.\n\n"
        "Fraud indicators include (but are not limited to):\n"
        "- Asking for PIN, IBAN, CVV, ΑΦΜ, ΑΜΚΑ, passwords or other sensitive data.\n"
        "- Using urgency words (e.g. 'άμεσα', 'τώρα', 'επείγον').\n"
        "- Requesting personal identification information (ταυτότητα, ημερομηνία γέννησης, πατρικό όνομα).\n"
        "- Pretending to be a bank, government, utility or other official organization.\n"
        "- Asking the user to click a link or open a website to enter credentials.\n\n"
        "Output rules:\n"
        "- Respond ONLY with a single word: YES (vishing) or NO (not vishing).\n"
        "- Do NOT output any explanation or probability.\n"
        "- If you are uncertain, respond with NO.\n\n"
        "Transcript:\n"
    )
    return instructions + text.strip() + "\n\nAnswer (YES or NO only):"


def build_vishing_probability_prompt(text: str) -> str:
    """
    Build a prompt asking the LLM to return ONLY JSON with vishing_probability.

    This is designed for calibrated probabilities and better separation:
    normal calls should be near 0, clear scams near 1.
    """
    return (
        "You are a fraud detection system specialized in detecting Greek vishing (voice phishing).\n\n"
        "You will be given a call transcript (Greek). Each new line indicates a speaker change.\n\n"
        "Output ONLY a single JSON object with:\n"
        '{\"vishing_probability\": <float between 0 and 1>}\n\n'
        "Scoring guidance:\n"
        "- 0.0–0.2: normal customer service / personal call (no sensitive data request, no impersonation, no urgency)\n"
        "- 0.3–0.6: suspicious but unclear\n"
        "- 0.7–1.0: very likely fraud (impersonation, urgency/pressure, sensitive data request, threats, link-to-enter-data)\n\n"
        "Transcript:\n"
        f"{text.strip()}\n"
    )


def build_domain_prompt(text: str) -> str:
    """
    Build a prompt asking the LLM to return ONLY JSON with a domain classification.

    Domains (exact labels):
    Bullying, Sexual Harassment, Hate Speech, Banking, Healthcare, Utilities,
    Education, Account Verification, Neutral, N/A
    """
    return (
        "Είσαι ταξινομητής θεματικού τομέα (domain) για τηλεφωνικές συνομιλίες.\n"
        "Θα σου δοθεί απομαγνητοφωνημένη συνομιλία (κάθε νέα γραμμή = αλλαγή ομιλητή).\n\n"
        "Διάλεξε ΑΚΡΙΒΩΣ ΕΝΑ domain από:\n"
        "Bullying | Sexual Harassment | Hate Speech | Banking | Healthcare | Utilities | "
        "Education | Account Verification | Neutral | N/A\n\n"
        "Κανόνες:\n"
        "- Banking: τράπεζα/λογαριασμός/κάρτα/χρέη/επενδύσεις.\n"
        "- Utilities: ρεύμα/νερό/αέριο/τηλεπικοινωνίες/πάροχος.\n"
        "- Healthcare: γιατροί/ραντεβού/φαρμακείο/ασφάλιση υγείας.\n"
        "- Education: σχολείο/μαθήματα/φοιτητικά/εκπαίδευση.\n"
        "- Account Verification: όταν το κύριο θέμα είναι ρητά επαλήθευση/επιβεβαίωση ταυτότητας/λογαριασμού.\n"
        "- Bullying / Hate Speech / Sexual Harassment: όταν το κύριο θέμα είναι εκφοβισμός, ρητορική μίσους, ή σεξουαλική παρενόχληση.\n"
        "- Neutral: γενική εξυπηρέτηση/παραγγελίες/πληροφορίες.\n"
        "- N/A: πολύ σύντομο ή άσχετο κείμενο.\n\n"
        "Απάντησε ΜΟΝΟ με JSON:\n"
        '{\"domain\": \"<one of the allowed domains>\"}\n\n'
        "Συνομιλία:\n"
        f"{text.strip()}\n"
    )


def build_probability_prompt(text: str) -> str:
    """
    Build a prompt asking the LLM to return ONLY JSON with vishing_probability.

    Ported from experiment_utils.build_prompt().
    """
    prompt = (
        "[INST] You are a classifier that estimates the probability that a call transcript "
        "represents a vishing (voice phishing) attempt.\n\n"
        "Instructions:\n"
        "- Read the following text.\n"
        "- Output ONLY a single JSON object with one key:\n"
        '  {\n'
        '    "vishing_probability": <float between 0 and 1>\n'
        "  }\n"
        "- Do not include any explanations or extra text.\n\n"
        "Transcript:\n"
        f"{text} [/INST]\n"
    )
    return prompt


_DEFAULT_CLASSIFICATION_PROMPT = (
    "[INST] Analyze the following call transcript segment (in Greek or English) and classify it.\n"
    "Provide a JSON response with 'category' and 'sentiment'.\n\n"
    "Instructions:\n"
    "1. 'category': [Sales, Support, Billing, Technical, Other]\n"
    "2. 'sentiment': [Positive, Neutral, Negative]\n\n"
    "Transcript:\n"
    "{text} [/INST]\n"
)


def build_classification_prompt(text: str) -> str:
    """
    Build a category/sentiment classification prompt.

    This is the default prompt used by the production classifier.
    For hot-reloadable prompts from file, use PromptLoader instead.
    """
    return _DEFAULT_CLASSIFICATION_PROMPT.replace("{text}", text)


class PromptLoader:
    """
    Hot-reloadable prompt loader from a text file.

    Checks file modification time and reloads on change.
    Falls back to default classification prompt if file is missing.

    Ported from the existing classifier's PromptLoader.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._cache: str | None = None
        self._last_mtime: float = 0
        self._fallback = _DEFAULT_CLASSIFICATION_PROMPT

    async def get_prompt(self, text: str) -> str:
        try:
            mtime = await asyncio.to_thread(
                lambda: os.path.getmtime(self.filepath)
            )
            if mtime > self._last_mtime or self._cache is None:
                logger.info(
                    f"Prompt file {self.filepath} changed or not loaded, reloading..."
                )
                content = await asyncio.to_thread(
                    lambda: open(self.filepath, "r").read()
                )
                self._cache = content
                self._last_mtime = mtime

            return self._cache.replace("{text}", text)
        except Exception as e:
            logger.warning(f"Error in PromptLoader.get_prompt: {e}", exc_info=True)
            return self._fallback.replace("{text}", text)
