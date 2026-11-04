# src/config.py
"""Single source of truth for tunable settings.

Values live in config.yaml at the project root so they can be changed without
touching Python. Anything the YAML doesn't specify falls back to the defaults
below, so the app still runs if the file is missing or PyYAML isn't installed.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_DEFAULTS = {
    "embedding": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "multilingual_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    "scoring": {
        "strong": 0.85,
        "good": 0.70,
        "partial": 0.55,
    },
    "matching": {
        # Calibrated by eval/run_eval.py — see the note in config.yaml.
        "covered": 0.36,
        "partial": 0.28,
        "reranker": {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "covered": -10.70,
            "partial": -11.20,
        },
    },
    "cleaning": {
        "stopwords_extra": [],
    },
    "skills": {
        "path": "data/skills_list.txt",
    },
    "llm": {
        "gemini": "gemini-flash-latest",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
    },
}


def _load_yaml():
    """Read config.yaml, or return {} if it can't be read for any reason."""
    try:
        import yaml
    except ImportError:
        return {}
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _merge(defaults, override):
    """Shallow per-section merge — a section in the YAML tops up, not replaces."""
    merged = {}
    for section, values in defaults.items():
        merged[section] = dict(values)
        if isinstance(override.get(section), dict):
            merged[section].update(override[section])
    return merged


_cfg = _merge(_DEFAULTS, _load_yaml())

EMBEDDING_MODEL = _cfg["embedding"]["model"]
EMBEDDING_DIM = int(_cfg["embedding"]["dimensions"])

# Loaded only when a document actually contains Arabic script. Empty string
# disables the route and leaves Arabic to the English model, which cannot
# represent it — see config.yaml for why that is a route and not a default.
MULTILINGUAL_MODEL = _cfg["embedding"].get("multilingual_model", "")

# Ordered high -> low; utils.score_band walks these.
SCORE_BANDS = {
    "strong": float(_cfg["scoring"]["strong"]),
    "good": float(_cfg["scoring"]["good"]),
    "partial": float(_cfg["scoring"]["partial"]),
}

MATCH_THRESHOLDS = {
    "covered": float(_cfg["matching"]["covered"]),
    "partial": float(_cfg["matching"]["partial"]),
}

_reranker = _cfg["matching"].get("reranker") or {}
RERANKER = {
    "model": (_reranker.get("model") or "").strip(),
    "covered": float(_reranker.get("covered", -10.70)),
    "partial": float(_reranker.get("partial", -11.20)),
}

STOPWORDS_EXTRA = list(_cfg["cleaning"]["stopwords_extra"])

LLM_MODELS = dict(_cfg["llm"])

SKILLS_PATH = PROJECT_ROOT / _cfg["skills"]["path"]


# Generated from an external source by tools/fetch_vocabulary.py. Kept
# separate from the hand-written list so the two can never be confused: the
# generated file is ~780 technology names nobody on this project curated, and
# the hand-written one holds the concepts no external source has a term for
# ("version control", "gradient boosting"). A vocabulary written by the same
# person who reads the failures can only ever contain what they thought of,
# which is how the first one scored 8/8 on the resume it came from and 2/7 on
# one it did not.
GENERATED_SKILLS_PATH = PROJECT_ROOT / "data" / "skills_generated.txt"


def _read_terms(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = (line.strip().lower() for line in f)
        return [line for line in lines if line and not line.startswith("#")]


#: Whether to fold the externally-generated vocabulary into the hand-written
#: one. OFF by default, and that default is a measurement rather than a
#: preference: on the 179-pair development set the generated vocabulary moved
#: F1 from 0.70 to 0.68 and the literal gate's accuracy from 93.9% to 89.3%.
#: See docs/EVALUATION.md, "What did not work".
USE_GENERATED_VOCABULARY = bool(_cfg["skills"].get("use_generated", False))


def load_skills():
    """Read the skill vocabulary. Blank lines and # comments are ignored."""
    terms = set(_read_terms(SKILLS_PATH))
    if USE_GENERATED_VOCABULARY:
        terms |= set(_read_terms(GENERATED_SKILLS_PATH))
    return sorted(terms)


SKILLS = load_skills()

ALIASES_PATH = PROJECT_ROOT / "data" / "skill_aliases.yaml"
GENERATED_ALIASES_PATH = PROJECT_ROOT / "data" / "skill_aliases_generated.yaml"


def load_aliases():
    """concept -> set of terms that count as evidence for it. See the YAML."""
    try:
        import yaml
    except ImportError:
        return {}
    merged = {}
    # Generated first so the hand-written file wins on any collision: it holds
    # the deliberate concept -> instance relations, which are a stronger claim
    # than a spelling synonym and should not be overwritten by one.
    paths = ((GENERATED_ALIASES_PATH, ALIASES_PATH)
             if USE_GENERATED_VOCABULARY else (ALIASES_PATH,))
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception:
            continue
        for concept, terms in raw.items():
            key = str(concept).lower()
            merged.setdefault(key, set()).update(str(t).lower() for t in (terms or []))
    return merged


SKILL_ALIASES = load_aliases()
