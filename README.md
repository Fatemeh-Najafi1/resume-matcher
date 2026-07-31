# Resume Matcher

Compare a resume against a job description. Returns a semantic match score, an
honest skills-gap report, and optional AI-written suggestions — **without ever
proposing a skill you don't have**.

## The no-fabrication guarantee

The gap report is produced by literal set difference over the two documents'
keywords. There is no model in that path, so there is no mechanism by which it
could invent a skill. The guarantee is structural, not a promise made to a
prompt. The optional LLM step is layered *on top* of that, never in place of it.

## Pipeline

```
Reader → Cleaner → Embedder → Comparator ─┐
                     └──────────────────→ Generator → report
```

| Stage | File | What it does |
|---|---|---|
| Reader | `src/Reader.py` | TXT / DOCX / PDF → plain text. Raises `UnreadableDocumentError` on a scanned PDF rather than returning empty text. |
| Cleaner | `src/Cleaner.py` | Normalize, extract GPA / years, redact contact details, lemmatize. Emits two views (see below). |
| Embedder | `src/Embedder.py` | `all-MiniLM-L6-v2` → 384-dim vector. Model is cached per process. |
| Comparator | `src/Comparator.py` | Cosine similarity between the two vectors, plus a plain-English verdict. |
| Generator | `src/Generator.py` | Keyword and skill gap report. No model involved. |

Optional, layered on top:

| Module | What it does |
|---|---|
| `src/Scraper.py` | Pull a job description straight from a posting URL. |
| `src/LLMAnalyzer.py` | Gemini / OpenAI / Anthropic analysis returning structured JSON. |
| `src/CompanyResearcher.py` | Company facts via DuckDuckGo, for cover-letter personalization. |

### Why Cleaner emits two views

`preprocess()` returns both `clean_text` and `keywords`, because the two
consumers want opposite things:

- **`clean_text`** — lemmatized, stopwords **kept**. Goes to the Embedder.
  MiniLM was trained on natural sentences; stripping "and"/"with"/"of"
  degrades the vector rather than sharpening it.
- **`keywords`** — stopwords **removed**. Goes to the Generator, where the
  comparison is a literal set difference and filler words would swamp the gap.

Lemmatization runs verb-first, then noun. `WordNetLemmatizer` defaults to
`pos="n"`, which leaves every verb form untouched — and resumes are written
almost entirely in past-tense verbs, so the default did essentially nothing.

## The two scores

The UI shows two numbers, and they can legitimately disagree:

- **Semantic similarity** — cosine similarity of the embeddings. Measures
  topic and vocabulary overlap, not whether you're qualified.
- **AI match score** — the LLM's own judgement after reading both documents.

A large gap between them is itself informative. They're labelled separately so
neither one's explanation is attached to the other's number.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then add whichever provider key you use
```

NLTK's `stopwords` and `wordnet` corpora download automatically on first run.

## Run

```bash
streamlit run app_streamlit.py           # web UI
python app.py --resume CV.pdf --job JD.txt   # CLI
```

## Configuration

`config.yaml` is the single source of truth for the embedding model, the score
bands, extra stopwords, and the skills-vocabulary path. `src/config.py` loads it
and falls back to built-in defaults if it's missing. The skill vocabulary itself
lives in `data/skills_list.txt`.

## Known limitations

- **Cosine similarity can flatter a buzzword-dense resume.** It measures
  topical closeness, not depth — it can't tell "used Python once" from "used
  Python daily for three years."
- **Skill matching is literal.** A skill phrased unusually, or absent from
  `data/skills_list.txt`, is simply not seen. Bare `go` and `r` are excluded
  from the vocabulary on purpose: one- and two-letter entries that are also
  ordinary English words fire on nearly every document.
- **Scanned PDFs are rejected, not OCR'd.** The failure is now explicit rather
  than silent, but adding OCR would be the real fix.
- **`openai` / `ollama` backends raise `NotImplementedError`.** The dispatch
  points exist so a backend can be added without touching the rest of the
  pipeline; nothing is wired up behind them yet.
