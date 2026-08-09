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


def load_skills():
    """Read the skill vocabulary. Blank lines and # comments are ignored."""
    if not SKILLS_PATH.exists():
        return []
    with open(SKILLS_PATH, "r", encoding="utf-8") as f:
        lines = (line.strip().lower() for line in f)
        return [line for line in lines if line and not line.startswith("#")]


SKILLS = load_skills()

ALIASES_PATH = PROJECT_ROOT / "data" / "skill_aliases.yaml"


def load_aliases():
    """concept -> set of terms that count as evidence for it. See the YAML."""
    try:
        import yaml
    except ImportError:
        return {}
    if not ALIASES_PATH.exists():
        return {}
    try:
        with open(ALIASES_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return {
        str(concept).lower(): {str(t).lower() for t in (terms or [])}
        for concept, terms in raw.items()
    }


SKILL_ALIASES = load_aliases()
