# Resume Matcher

A Streamlit app that compares a resume against a job description using NLP embeddings, computes a semantic similarity score, and identifies skill gaps — without fabricating skills you don't have.

🔗 **[Live Demo](https://resume-m-nkdtv7rpdzsjvtktlqx5zt.streamlit.app/)**

---

## The Problem This Solves

Most resume tools either score your resume blindly or suggest adding skills you don't actually have. This tool takes a different approach: it tells you how well you genuinely match a role, surfaces the specific keywords and skills you're missing, and explicitly flags which ones you should develop before claiming them — not add dishonestly.

**The guarantee is structural, not a promise made to a prompt.** The gap report is a literal set difference over the two documents. There is no model in that code path, so there is no mechanism by which it could invent a skill. The optional LLM analysis is layered *on top* of that, never in place of it.

---

## What It Does

- Accepts a resume and job description in **TXT, DOCX, or PDF** format, or scrapes the posting straight from a **URL**
- Cleans and preprocesses both documents (normalization, lemmatization, noise removal)
- Extracts structured metadata: GPA, years of experience, contact info
- Embeds both documents using **sentence-transformers** (`all-MiniLM-L6-v2`)
- Computes **cosine similarity** to produce a match score (0–100%)
- Breaks the posting into individual **requirements** and matches each against individual resume lines, reporting coverage and the specific line that evidences each one
- Identifies keywords present in the job description but missing from your resume, ranked by how often the posting repeats them
- Suggests which terms to add (if you have the experience) and which to develop first
- Optional **AI analysis** (Gemini / OpenAI / Anthropic) and automatic company research for cover letters

---

## Architecture

```
Input (TXT / DOCX / PDF / URL)
        ↓
    Reader        → loads resume and job description
        ↓
    Cleaner       → normalizes, lemmatizes, extracts GPA / YoE
        ↓
    Embedder      → sentence-transformers (all-MiniLM-L6-v2)
        ↓
    Comparator    → cosine similarity + score interpretation
        ↓
    Generator     → keyword gap analysis + match report
        ↓
  Requirements    → splits the posting into discrete requirements
    + Matcher     → matches each against individual resume bullets
        ↓
  Streamlit UI    → score, coverage, evidence, suggested updates
```

---

## Design Notes

Three decisions that aren't obvious from the code:

**Documents are chunked before embedding.** `all-MiniLM-L6-v2` accepts 256 word-piece tokens and silently discards the rest. A typical resume is ~630 tokens, so more than half of it was never being read and the score described only the opening of each document. Text is now split into overlapping windows and mean-pooled. Averaging blurs a document covering several topics, but reading all of it blurred beats reading 40% of it sharply.

**The Cleaner emits two views of each document.** `clean_text` keeps stopwords and goes to the Embedder — MiniLM was trained on natural sentences, so stripping "and"/"with"/"of" degrades the vector rather than sharpening it. `keywords` has stopwords removed and goes to the Generator, where the comparison is a literal set difference and filler words would swamp the result.

**Some things are checked literally rather than judged by similarity.** Cosine answers *"is this similar work?"* well and *"do they know this specific thing?"* badly — "experience using frameworks such as PyTorch" scores **0.57** against a resume line reading `scikit-learn, NLTK`, high enough that similarity alone reported PyTorch as covered when the resume never mentions it. Three classes are therefore decided by inspection, not by vectors:

| Class | Rule |
|---|---|
| Named technology | Is the word actually there, or an alias of it? |
| Degree level | A Master's does not satisfy a PhD |
| Duration demand | "1 year of ML research" is not proven by listing scikit-learn — capped at partial |

**And some things aren't decided at all.** "Attention to detail" cannot be settled by a resume, so the app shows the closest matching line and withholds a verdict. See Evaluation for why that turned out to be the honest choice rather than a cop-out.

**The thresholds are measured, not chosen.** See Evaluation below.

---

## Evaluation

`data/eval/pairs.jsonl` holds 124 labelled (requirement, resume-bullet) pairs. `python -m eval.run_eval` scores the matcher against them and sweeps thresholds; `--errors` prints what it got wrong.

Every threshold in `config.yaml` comes from that sweep. The first version of this project set them by eye, and the eval showed how badly: the original 0.55 scored **precision 1.00 at recall 0.20** — it missed four out of five genuine matches.

5-fold cross-validated, threshold tuned on train folds only. "Answerable" excludes requirements about personal qualities, which the app declines to rule on (see below):

| Scorer | Precision | Recall | Answerable P | Answerable R |
|---|---|---|---|---|
| bi-encoder + literal gate | 0.90 | 0.73 | 0.90 | 0.78 |
| **cross-encoder + literal gate** | **0.91** | **0.82** | **0.93** | **0.89** |
| NLI entailment + literal gate | 0.89 | 0.82 | 0.93 | 0.89 |

For comparison, the original hand-picked threshold scored **precision 1.00 at recall 0.20**.

Precision is weighted over recall throughout, because a false "covered" tells someone they meet a requirement they do not — the exact failure this project exists to prevent.

### Why some requirements get no verdict

"Attention to detail" and "ability to thrive in a fast-paced environment" cannot be settled by a resume. What made that concrete was labelling the eval set and catching *my own* inconsistency: I marked mentoring as evidence of "team leadership" but not of "works in a fast-paced environment". If the person writing the labels can't be consistent, a model scored against them is being graded on a coin flip. Those requirements now show the closest matching line, no verdict, and are excluded from the coverage figure.

### Two findings from building this

- **The first eval set gave the wrong answer.** Built only from the strong senior resume in `resume.txt`, it over-represented cases where a match exists, and concluded a recalibrated bi-encoder beat the cross-encoder. Running the app on a real junior-resume-vs-senior-role comparison produced false positives the benchmark had not predicted. Pairs 91–124 add that distribution and the ranking reversed. **Eval set construction mattered more than model choice.**
- **The alias map silently did nothing at first.** Its keys weren't part of the skill vocabulary, so "gradient boosting" was never recognised as a skill, the gate never engaged, and the aliases never fired. Fixing that one line moved recall from 0.71 to 0.89.

### How much to trust these numbers

Two forces pull in opposite directions, and neither cancels the other:

- **Pessimistic:** the set is a curated collection of hard cases, not a random sample. Real traffic contains many easy pairs this set deliberately omits.
- **Optimistic, and more serious:** the alias map and the vocabulary fix were written *after* inspecting which pairs the matcher got wrong. That is test-set contamination.

So treat the **ranking** of scorers as sound and the **absolute values** as unconfirmed. Establishing them properly needs ~100 freshly labelled pairs held out and unseen until the end. That is the top open item.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| UI | Streamlit |
| NLP / Embeddings | sentence-transformers, NLTK |
| File parsing | pdfplumber, docx2txt |
| Scraping | requests, BeautifulSoup |
| LLM (optional) | Gemini, OpenAI, Anthropic |
| ML | scikit-learn, NumPy |
| Config | YAML |

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Fatemeh-Najafi1/resume-matcher.git
cd resume-matcher

# Install dependencies
pip install -r requirements.txt

# Optional: add an API key for the AI analysis
cp .env.example .env

# Run the Streamlit app
streamlit run app_streamlit.py
```

Or use the CLI version:
```bash
python app.py --resume your_resume.txt --job job_description.txt
```

NLTK's `stopwords` and `wordnet` corpora download automatically on first run. The app works without an API key — you lose the AI analysis, not the matching.

---

## Project Structure

```
resume-matcher/
├── app.py                  # CLI entry point
├── app_streamlit.py        # Streamlit UI
├── config.yaml             # Model, thresholds, stopwords, skills path
├── requirements.txt
├── src/
│   ├── config.py           # Loads config.yaml, with built-in defaults
│   ├── Reader.py           # File loader (TXT, DOCX, PDF)
│   ├── Cleaner.py          # Text preprocessing pipeline
│   ├── Embedder.py         # Chunked, cached embeddings
│   ├── Comparator.py       # Similarity scoring + insight
│   ├── Generator.py        # Keyword gap analysis + report
│   ├── Requirements.py     # Splits a posting into discrete requirements
│   ├── Matcher.py          # Requirement ↔ resume-bullet matching
│   ├── Reranker.py         # Optional cross-encoder reranking
│   ├── Scraper.py          # Job posting URL scraping
│   ├── LLMAnalyzer.py      # Gemini / OpenAI / Anthropic analysis
│   ├── CompanyResearcher.py# Company facts for cover letters
│   └── utils.py            # Shared score bands, skill and degree lookup
├── docs/
│   └── ENGINEERING-LOG.md  # Decisions, findings and their reasoning
├── eval/
│   └── run_eval.py         # Threshold calibration and scorer comparison
└── data/
    ├── skills_list.txt     # Skill vocabulary
    ├── skill_aliases.yaml  # Git ⇒ version control, XGBoost ⇒ gradient boosting
    └── eval/pairs.jsonl    # 124 labelled requirement/bullet pairs
```

---

## Configuration

`config.yaml` is the single source of truth for the embedding model, score bands, match thresholds, stopwords and the skills-vocabulary path. `src/config.py` loads it and falls back to built-in defaults if it's missing.

---

## Job Posting URLs

| Source | Status |
|---|---|
| LinkedIn | ✅ via LinkedIn's public guest endpoint, no login required |
| Greenhouse, Lever, Workday, SmartRecruiters, Ashby | ✅ |
| Company career pages | ✅ tries schema.org `JobPosting` data first, then page structure |
| Indeed, Glassdoor | ⚠️ Cloudflare bot check — the app detects the block and asks you to paste instead |

---

## Known Limitations

- **Cosine similarity can flatter a buzzword-dense resume.** It measures topical closeness, not depth — it can't tell "used Python once" from "used Python daily for three years."
- **Skill matching is literal.** A skill phrased unusually, or absent from `data/skills_list.txt`, is simply not seen. Bare `go` and `r` are excluded on purpose: one- and two-letter entries that are also ordinary English words fire on nearly every document.
- **Requirements too general to check literally** ("integrate models into existing strategies") fall back to similarity, which is the weakest part of the system. Those rows are labelled in the UI so you can judge them yourself.
- **Requirements about personal qualities get no verdict at all**, by design. If a posting is mostly soft skills, the coverage figure will be based on only a few requirements — check the "Your call" count before reading much into it.
- **The evaluation numbers are not yet trustworthy as absolutes.** See "How much to trust these numbers".
- **Scanned PDFs are rejected, not OCR'd.** The failure is explicit rather than silent, but OCR would be the real fix.
- **The `openai` / `ollama` embedding backends raise `NotImplementedError`.** The dispatch points exist so a backend can be added without touching the rest of the pipeline; nothing is wired behind them yet.

---

## Roadmap

- [x] Scrape job descriptions directly from LinkedIn or company websites
- [x] Skill gap categorization (have it vs. need to learn it)
- [x] LLM-powered analysis and bullet rewriting (Gemini / OpenAI / Anthropic)
- [x] Requirement-level coverage with per-requirement evidence
- [x] Labelled evaluation set and threshold calibration
- [x] Cross-encoder reranking, measured against the benchmark before shipping
- [ ] Fresh held-out labelled set to confirm the evaluation numbers
- [ ] Interactive resume tailoring that asks before claiming any skill
- [ ] Multi-resume comparison against a single job description
- [ ] Export tailored resume as PDF or DOCX
- [ ] Local LLM support via Ollama

---

## Author

**Fatemeh Najafi** — [GitHub](https://github.com/Fatemeh-Najafi1) · [LinkedIn](https://www.linkedin.com/in/fatemeh-najafi-797555382)
