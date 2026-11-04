# Resume Matcher

Matches a resume against a job posting requirement by requirement, shows the
exact line that evidences each one, and — the part that distinguishes it —
**separates "your resume doesn't mention this" from "your resume rules this
out".**

🔗 **[Live Demo](https://resume-m-nkdtv7rpdzsjvtktlqx5zt.streamlit.app/)**

---

## The distinction this is built around

Every tool in this category collapses "not covered" into one bucket. Two very
different things land in it:

| | What it means | What should happen |
|---|---|---|
| **❓ Unsupported** | The resume is silent. The candidate may well have it and simply not written it down. | Ask. Or add it to the resume. |
| **🚫 Contradicted** | The resume states something incompatible — a bachelor's against a PhD demand, two years against eight. | This one is real. |

Conflating them is how a qualified person gets filtered out for an omission
rather than a shortfall. Keeping them apart costs almost nothing and changes
what the output is *for*: a list of questions to ask, not a verdict to accept.

Two asymmetries keep the `contradicted` bucket honest, both enforced in
[`src/Verdict.py`](src/Verdict.py):

1. **Absence is never contradiction.** A resume that never mentions Kafka is
   unsupported on Kafka. Nothing about an unwritten line proves an unheld
   skill.
2. **Contradiction needs a stated value on both sides.** The requirement must
   name a floor and the resume must state its own value. If either is silent,
   it falls back to unsupported.

Both push errors toward "ask" and away from "reject".

---

## What it does

- Reads **TXT, DOCX or PDF**, or scrapes the posting from a **URL**
- Splits the posting into discrete requirements and matches each against
  individual resume lines
- Reports per-requirement coverage **with the specific line as evidence**
- **Abstains** on personal qualities a resume cannot settle, rather than
  guessing
- Handles **English and Arabic**, including cross-language matching
- **Checks every AI-suggested rewrite** for invented skills, invented numbers
  and inflated seniority claims
- Optional LLM analysis (Gemini / OpenAI / Anthropic) and company research

---

## Evaluation

Full protocol: **[docs/EVALUATION.md](docs/EVALUATION.md)**. The short version
is that there are three sets — one to tune on, two to be measured by — and the
second kind never informs a decision.

```bash
python -m eval.run_eval      # development set: tune and compare
python -m eval.holdout --all # held-out sets: measure
```

### The baseline ladder, with cost

Every rung measured on the same pairs. The bottom rung has no model at all,
which is the point: anything that cannot beat word overlap has no business in
the pipeline.

**Development set (124 pairs) — tuned against, so read as an upper bound:**

| Scorer | P | R | F1 | ms/pair | disk |
|---|---|---|---|---|---|
| Keyword overlap (Jaccard) | 0.91 | 0.78 | 0.84 | 0.0 | — |
| BM25 | 0.91 | 0.78 | 0.84 | 0.0 | — |
| Skills mapping (literal gate) | 0.96 | 0.51 | 0.67 | 0.5 | — |
| Semantic search (bi-encoder) | 0.90 | 0.75 | 0.82 | 95 | 183MB |
| **Ranking (cross-encoder)** | **0.93** | **0.80** | **0.86** | 98 | 184MB |
| Entailment (NLI) | 0.91 | 0.76 | 0.83 | 165 | 331MB |

> **On the timings.** The development set runs first, so its millisecond
> column carries the one-off cost of loading each model from disk. Compare
> timings *within* a table, not across them — the held-out table below is
> closer to steady-state throughput, which is why its bi-encoder figure is a
> third of the one above for identical work.

**Held-out set (60 pairs, different persona, never tuned against):**

| Scorer | P | R | F1 | ms/pair |
|---|---|---|---|---|
| Keyword overlap (Jaccard) | 0.82 | 0.47 | 0.60 | 0.0 |
| BM25 | 0.82 | 0.47 | 0.60 | 0.0 |
| Skills mapping | 0.90 | 0.30 | 0.45 | 0.5 |
| **Semantic search** | 0.81 | 0.57 | **0.67** | 29 |
| Ranking (cross-encoder) | 0.84 | 0.53 | 0.65 | 182 |
| Entailment (NLI) | 0.76 | 0.53 | 0.63 | 174 |

### Three findings, and none of them is flattering

**1. The contamination was real, and this is its size.** The previous README
admitted its numbers were tainted — the alias map and skill vocabulary were
both repaired after inspecting which pairs failed. It could not say by how
much. Now it can: **F1 0.86 on the set that was tuned, 0.65 on the set that
was not.** Synonym accuracy went 8/8 → 2/7, paraphrase 12/14 → 3/7.

The cause is precise. The alias map contains the synonyms the development set
happens to use — Git for version control, XGBoost for gradient boosting. It
has never heard of Terraform meaning infrastructure-as-code or Parquet meaning
columnar storage. The development-set synonym score was never measuring
synonym handling; it was measuring how thoroughly the map had been fitted to
one resume.

The fix is not more aliases. Adding data-engineering terms now would re-fit the
map to the held-out set and burn it. The fix is a vocabulary nobody hand-fitted
— ESCO's alternative labels — which is the top open item.

**2. The cross-encoder's advantage does not survive the move.** It wins on the
development set (F1 0.86 vs 0.82) and loses on the held-out set (0.65 vs 0.67)
while costing **six times the latency**. This is the second time this ranking
has reversed when the evaluation distribution changed; the first is written up
in [the engineering log](docs/ENGINEERING-LOG.md).

The honest reading is not "the bi-encoder is better". It is that **this
benchmark cannot settle the question**, and the shipped default has
deliberately *not* been changed on the strength of a held-out number — doing so
would convert the held-out set into a development set. Settling it needs a
third set.

**3. A model with no model beats most of them.** Jaccard overlap scores F1 0.60
held-out, in microseconds, with nothing to download. The best transformer
manages 0.67. That gap is what 183MB and 29ms per pair actually buys, and
reporting it is the only way a reader can judge whether it is worth it.

### Selective prediction

The matcher abstains, so a single precision figure describes a system that
does not exist. What it can answer, and how often it is right when it does:

| | Development | Held-out |
|---|---|---|
| Answering its most confident 30% | 94.6% correct | 88.9% correct |
| Answering 70% | 94.2% | 78.6% |
| Answering everything | 89.5% | 71.7% |
| AURC (vs shuffled baseline) | 0.046 (0.102) | 0.197 (0.280) |
| Abstain ECE | 0.005 | 0.178 |
| Coverage under 10% error | 98% | 35% |

The confidence signal is informative on both — it beats its shuffled baseline
either way, so the ordering means something. But **calibration does not
transfer**: ECE 0.005 in-domain against 0.178 out of it. In plain terms, the
confidence numbers are trustworthy as a *ranking* and should not be shown to a
user as probabilities out of domain. They are not.

---

## Arabic

Postings in the Gulf are often written in Arabic while technical resumes are
usually written in English, so a requirement and the evidence for it routinely
sit in different scripts.

**Measured on 44 bilingual pairs:**

| Scorer | P | R | F1 | ms/pair | disk |
|---|---|---|---|---|---|
| Keyword overlap | 0.93 | 0.57 | 0.70 | 0.1 | — |
| Skills mapping (literal gate) | 0.92 | 0.52 | 0.67 | 0.8 | — |
| Semantic search (English model) | 0.82 | 0.61 | 0.70 | 87 | 183MB |
| **Semantic (multilingual)** | **0.89** | **0.70** | **0.78** | 420 | 480MB |
| Ranking (cross-encoder, English) | 0.75 | 0.65 | 0.70 | 260 | 184MB |

That gap is not a surprise — `all-MiniLM-L6-v2` has no Arabic in its training
data and cannot represent an Arabic requirement at all. It does not fail when
handed one; it returns a vector, and the vector is noise. So the encoder is
**routed by script**, not chosen by benchmark:

- Any Arabic in either document → multilingual encoder (480MB, 128-token
  window), **and the cross-encoder reranker is switched off entirely**. The
  configured reranker is English-only, and because it *overrides* cosine
  wherever it runs, leaving it on threw away the multilingual encoder's work
  and replaced it with confident noise. Routing the embedder without routing
  the reranker was a real bug, found by running this set.
- Otherwise → the English model (90MB, 256-token window), which is faster and
  smaller

Note the first row of that table. On Arabic, plain word overlap beats every
English-model rung, because the only thing those rungs can match on is the
Latin tokens that survive in a mixed-script line. That is a useful
sanity-check on what "semantic" matching is actually doing cross-lingually.

Measured on the path the app actually takes (multilingual encoder, reranker
off): **79.5% accurate answering everything, 84.6% answering its most
confident 30%**, AURC 0.173 against a 0.202 shuffled baseline — so the
confidence ordering carries real information in Arabic too. Calibration does
not survive the language change, though: Abstain ECE 0.118, which is why the
number is not shown to users as a probability.

Per direction: Arabic requirement → English evidence **17/20**, Arabic →
Arabic 9/11, English → Arabic 9/13. The weakest direction is an English
requirement against an Arabic resume line, which is the rarer case in
practice.

Getting this block right required fixing the harness as well as the matcher.
It had been reporting selective prediction for the cross-encoder on the
bilingual set — a scorer the app disables for Arabic — and that phantom
measurement showed the confidence signal as *no better than chance*. The
reported conclusion was wrong because the thing being measured was not the
thing being shipped.

The chunker derives its stride from the loaded model's window rather than
assuming 256, because a constant that is correct for one model will outlive
that model.

Three further things were needed to make the literal half work, and each was a
silent failure before:

- **Orthographic folding.** Arabic writes the same word several legal ways —
  alef with or without hamza (أ/إ/آ/ا), yeh as ي or ى, teh marbuta as ة or ه,
  optional diacritics, tatweel stretching (تحليـــل). Without folding, a match
  simply never fires and nothing reports that it didn't.
- **Unicode word boundaries.** The old pattern `(?<![a-z0-9])` does nothing in
  Arabic — an Arabic letter is not in `[a-z0-9]`, so the lookaround passed and
  terms matched inside longer words.
- **Arabic degree markers.** The degree list held only English, so every Arabic
  degree requirement bypassed the gate and fell through to cosine, which
  cheerfully returned "covered" for a master's demand against a resume with no
  master's. Confident, silent and wrong.

**Honest caveat.** Those three fixes were made *after* the bilingual set had
been run, which means the post-fix bilingual numbers are no longer a clean
held-out measurement for the degree and skill categories. They are reported as
what they are: a before-and-after on a set that has now been touched.
Confirming them needs fresh pairs. The fixes themselves were justified by
reading the code — a marker list containing no Arabic cannot match Arabic —
rather than by the benchmark, which is the only reason they were made at all.

---

## The no-fabrication guarantee, and where it stops

The gap report is a literal set difference. **There is no model in that code
path**, so there is no mechanism by which it could invent a skill. That part of
the guarantee is structural rather than promised.

The optional LLM panel is different: it rewrites bullets, and its instruction
not to invent anything is a sentence in a prompt. Prompts are not load-bearing.
So every rewrite is checked after generation ([`src/Fidelity.py`](src/Fidelity.py)):

| Check | Action |
|---|---|
| A skill the resume never mentions | **Rejected** and shown marked |
| A number the resume never states | **Rejected** — the invented "improved by 40%" |
| A seniority verb the original didn't use | **Warned** — "led", "owned", "architected" |
| A skill moved from elsewhere in the resume | **Flagged** for the user's eye |

Nothing is silently dropped. A rejected rewrite is still displayed, marked and
explained, because the useful information is not "here are three suggestions"
but "this tool just tried to put a number in your mouth".

The third row is a lexical proxy for something genuinely unsolved: a rewrite
can imply seniority without using any of those words, and nothing here catches
that.

---

## Architecture

```
Input (TXT / DOCX / PDF / URL)
        ↓
    Reader      → loads resume and posting
        ↓
    Cleaner     → normalises, lemmatises, extracts GPA / years
        ↓
 Requirements   → splits the posting into discrete requirements
        ↓
   Embedder     → routed by script: English or multilingual
        ↓
    Matcher     → requirement × bullet
        ├── Verdict  → literal gates: skills, degrees, durations
        └── Reranker → optional cross-encoder over the shortlist
        ↓
   Streamlit    → per-requirement verdict, evidence line, abstentions
```

| Module | Does |
|---|---|
| [`src/Verdict.py`](src/Verdict.py) | The verdict vocabulary and the silence/mismatch rule |
| [`src/Matcher.py`](src/Matcher.py) | Requirement ↔ bullet matching, literal gate, evidence selection |
| [`src/Fidelity.py`](src/Fidelity.py) | Post-generation check on LLM rewrites |
| [`src/Embedder.py`](src/Embedder.py) | Chunked, cached, script-routed embeddings |
| [`eval/metrics.py`](eval/metrics.py) | Risk–coverage, AURC, Abstain ECE |
| [`eval/scorers.py`](eval/scorers.py) | The six-rung baseline ladder, timed |
| [`eval/holdout.py`](eval/holdout.py) | Held-out measurement; fits thresholds on dev only |

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py          # the app
python app.py --resume cv.txt --job jd.txt   # CLI
python -m pytest tests/ -q              # 32 tests
python -m eval.holdout --all            # every evaluation set
```

Works without an API key — you lose the AI panel, not the matching. NLTK
corpora download on first run. Everything is CPU-only.

Dependencies are **pinned** to the versions the numbers above were measured
against; they were unpinned until recently, which meant every rebuild of the
live demo resolved fresh.

---

## Known limitations

- **The held-out numbers are the honest ones, and they are not great.** F1 0.65–0.67
  out of domain. Anything quoting 0.86 is quoting the contaminated figure.
- **All three sets were labelled by one person**, so the inter-annotator
  agreement ceiling is unknown. On personal qualities the author caught
  themselves labelling inconsistently, which is why the app abstains there.
- **All three sets are curated hard cases**, not samples of real traffic. The
  numbers are pessimistic about the average posting and silent about the tail.
- **The alias map does not generalise** — demonstrated above, not suspected.
- **No fairness audit has been run.** The deterministic rungs should show
  near-exact invariance under name perturbation, which is a claim no LLM scorer
  can make. Untested, so unclaimed.
- **44 bilingual pairs** is enough to catch a gross failure and not enough to
  compare models with confidence.
- **Scanned PDFs are rejected, not OCR'd.** Explicit failure, but OCR is the
  real fix.
- **The `openai` / `ollama` embedding backends raise `NotImplementedError`.**
  Dispatch points exist; nothing is behind them.

---

## Roadmap

- [x] Requirement-level coverage with per-requirement evidence
- [x] Labelled evaluation set and threshold calibration
- [x] Cross-encoder reranking, measured before shipping
- [x] Held-out set — and the contamination quantified rather than confessed
- [x] Risk–coverage, AURC and calibration reporting
- [x] Separate "unsupported" from "contradicted"
- [x] Arabic / English cross-language matching
- [x] Programmatic fabrication check on LLM rewrites
- [ ] ESCO alternative labels to replace the hand-maintained alias map
- [ ] A second annotator, to establish the agreement ceiling
- [ ] Counterfactual fairness audit
- [ ] OCR for scanned PDFs

---

## Author

**Fatemeh Najafi** — [GitHub](https://github.com/Fatemeh-Najafi1) · [LinkedIn](https://www.linkedin.com/in/fatemeh-najafi-797555382)
