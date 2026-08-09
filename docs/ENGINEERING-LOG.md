# Engineering Log

Why this project is built the way it is. Written for two readers: anyone picking
the codebase up later, and the author revising before an interview.

Entries record the *reasoning*, including the times the reasoning turned out to
be wrong. The wrong turns are kept on purpose — they are the parts worth
explaining out loud.

---

## The one-line version

A resume/job-description matcher whose defining constraint is that **it must
never claim you have a skill you don't**. Every architectural decision below
follows from that constraint.

---

## 1. The no-fabrication guarantee is structural

The keyword gap report is a literal set difference between two documents. There
is no model in that code path, so there is no mechanism by which it could invent
a skill. This matters because the alternative — instructing an LLM not to
fabricate — is a request, not a guarantee.

The LLM analysis is layered *on top* of that, never in place of it.

**If asked why it matters:** during testing, Gemini claimed the candidate hadn't
mentioned cross-validation when the resume said "Random Forest classification
with cross-validation". The deterministic layer does not make that class of
mistake.

---

## 2. Bugs found by reading the code

These were all live in the original version.

| Bug | Why it mattered |
|---|---|
| **Embeddings silently truncated at 256 tokens** | `all-MiniLM-L6-v2` drops everything past its window with no warning. A 633-token resume was read at 40%. The similarity score described only the opening of each document. |
| **Stopword removal never ran** | The NLTK corpus was loaded, then `Cleaner` was constructed with an empty list. |
| **Lemmatisation did almost nothing** | `WordNetLemmatizer` defaults to `pos="n"`, leaving every verb untouched. Resumes are mostly past-tense verbs, so "managed" never became "manage". |
| **Skill extraction always returned `[]`** | It matched against a list that was never populated. |
| **`config.yaml` was never read** | Thresholds lived in four places with different values. |
| **Unreachable dispatch branches** | `Comparator` nested `if model == "basic"` inside itself; other backends returned `None` instead of raising. |
| **Every scrape error said "HTTP error unknown"** | `if e.response` is falsy for *any* error status in `requests`, because `Response.__bool__` returns `self.ok`. Needs `is not None`. |
| **`.env` was untracked but not ignored** | One `git add -A` from publishing a live API key. |

**Fix for truncation:** split into overlapping 150-word windows, embed each,
length-weighted mean. Averaging blurs a multi-topic document — but reading 100%
of it blurred beats reading 40% of it sharply.

---

## 3. Why the Cleaner emits two views of each document

`preprocess()` returns both `clean_text` and `keywords`, because the two
consumers want opposite things.

- **`clean_text`** — lemmatised, stopwords **kept** → the Embedder. MiniLM was
  trained on natural sentences; stripping "and"/"with"/"of" degrades the vector
  rather than sharpening it.
- **`keywords`** — stopwords **removed** → the Generator, where the comparison
  is a literal set difference and filler words would swamp the result.

This is a better answer than "we remove stopwords", and it is the honest
description of what the code does.

---

## 4. Document-level scoring was replaced by requirement-level matching

A single cosine over two whole documents produces one number that cannot be
acted on. 52.3% tells you nothing about *which* experience counted.

Now: split the posting into discrete requirements, split the resume into
individual lines, and score every requirement against every line. That yields a
coverage figure, and the specific line that evidences each requirement.

It also sidesteps truncation rather than mitigating it — a single requirement
and a single bullet both sit far inside the 256-token window.

---

## 5. Cosine similarity is the wrong tool for "do they know X?"

Measured, not assumed:

> `"experience using frameworks such as PyTorch"` scores **0.57** against
> `"Frameworks & Libraries: Flask, Streamlit, Pandas, NumPy, scikit-learn, NLTK"`

That clears any sensible threshold. Embedding space puts all ML tooling in the
same neighbourhood, so similarity cannot tell PyTorch from scikit-learn.

So three classes of requirement are decided by **inspection instead of vectors**:

- **Named technology** — is the word there, or an alias of it?
- **Degree level** — a Master's does not satisfy a PhD (ordinal, not fuzzy)
- **Duration** — "1 year of ML research" is not proven by listing scikit-learn;
  capped at partial

Cosine only rules on requirements too general to check literally.

---

## 6. The evaluation set, and how it misled me twice

Before this, every threshold was chosen by eyeballing one resume against one
posting. `data/eval/pairs.jsonl` (124 labelled pairs) and `eval/run_eval.py`
replaced that with measurement.

**First finding — the shipped threshold was badly wrong.** 0.55 scored
precision 1.00 at **recall 0.20**. It was right whenever it spoke, and it barely
spoke.

**Mistake 1 — a biased test set.** The first 90 pairs were built entirely from
the strong senior resume in `resume.txt`, so they over-represented cases where a
match exists. On that set a recalibrated bi-encoder beat the cross-encoder, and
I concluded no reranker was needed. Running the app on a real junior-resume-vs-
senior-role comparison immediately produced false positives the benchmark had
not predicted — coursework titles matching "enhance model architecture design".
Pairs 91–124 add that distribution and **the ranking reversed**.

> Eval set construction mattered more than model choice. This is the single most
> useful thing this project taught.

**Mistake 2 — labels I could not keep consistent.** In the soft-skill category I
had marked mentoring as evidence of "team leadership" but not of "works in a
fast-paced environment". Those are coin flips. A model scored against them is
being graded on noise. Rather than tune to my own inconsistency, the app now
**declines to rule** on personal-quality requirements and shows the closest
evidence instead.

**A silent no-op.** The alias map (`Git ⇒ version control`) initially changed
nothing: its keys were not in the skill vocabulary, so "gradient boosting" was
never recognised as a skill, the gate never engaged, and the aliases never
fired. One line. Recall 0.71 → 0.89.

---

## 7. Scorers compared

5-fold cross-validated, threshold tuned on train folds only, answerable
requirements only:

| Scorer | Precision | Recall |
|---|---|---|
| Original hand-picked threshold | 1.00 | 0.20 |
| bi-encoder + literal gate | 0.90 | 0.78 |
| **cross-encoder + literal gate** ← ships | **0.93** | **0.89** |
| NLI entailment + literal gate | 0.93 | 0.89 |
| Ensemble (logistic regression over all three) | 0.93 | 0.84 |

**Why the cross-encoder wins:** a bi-encoder embeds the two texts separately, so
they never see each other. A cross-encoder reads them together and attends
across both.

**Why not NLI**, despite tying: it fixes the synonym cases the cross-encoder
misses (1/8 → 6/8 — Docker evidencing containerisation, Git evidencing version
control) but the model is substantially larger for no net gain. It stays in
`eval/run_eval.py` so the comparison reproduces.

**Why not the ensemble:** it did not beat the single cross-encoder. Worth
recording as a negative result rather than quietly dropping.

---

## 8. Honest state of the numbers

Two forces, neither cancelling the other:

- **Pessimistic** — the set is curated hard cases, not a random sample. Real
  traffic has many easy pairs it deliberately omits.
- **Optimistic, and more serious** — the alias map and vocabulary fix were
  written *after* inspecting which pairs failed. That is **test-set
  contamination**.

Therefore: the **ranking** of scorers is sound; the **absolute values** are
unconfirmed. Confirming them needs ~100 freshly labelled pairs, held out and
unseen until the end. This is the top open item.

I stopped at recall 0.89 rather than closing the last point. Reading the five
remaining failures and patching for them would produce a number that is fiction.

---

## 9. Scraping: what is achievable and what isn't

| Source | Status |
|---|---|
| LinkedIn | Works — its public guest endpoint serves posting content without a login; `/jobs/view/<id>` alone redirects to a sign-in wall |
| Greenhouse, Lever, Workday, SmartRecruiters, Ashby | Works |
| Company career pages | Works — tries schema.org `JobPosting` data first |
| **Indeed, Glassdoor** | **Blocked.** Cloudflare returns 403 with `cf-mitigated: challenge` regardless of headers |

Only a real browser clears a Cloudflare JS challenge. That was deliberately not
pursued: on a deployed app the request comes from a datacenter IP, which
Cloudflare treats far more harshly, so the cost would buy nothing. The app
detects the block and degrades to a clear manual-paste path.

> Detecting a block and degrading honestly is a better answer than a brittle
> scraper that breaks the week Cloudflare tightens a rule.

---

## 10. Operational notes worth remembering

- **Listing a model is not proof you can call it.** `gemini-2.5-flash` appears
  in `list_models()` and still 404s with *"no longer available to new users"*.
  `gemini-2.0-flash` returns 429 with a free-tier quota of **0**. Only
  `gemini-flash-latest` worked. Model IDs live in `config.yaml` for this reason.
- **`temperature=0`** on all providers, so re-running the same inputs gives the
  same requirements.
- **`.env` does not exist on Streamlit Cloud** — it's gitignored. Keys fall back
  to `st.secrets`, or every visitor sees "no API key".
- **The model reloaded on every interaction** before caching: 7.21s → 0.08s.
- **Results were lost on any widget click** before `session_state`: Streamlit
  re-runs the script, the button goes false, and the whole analysis vanished.

---

## Open items

1. **Fresh held-out labelled set** — the only way to confirm the numbers above.
2. **Second annotator on the existing 124** — disagreement rate is the ceiling
   no model can beat.
3. **Interactive tailoring** — ask "do you have this?" before any rewrite, with
   a programmatic verifier rejecting rewrites that introduce unconfirmed skills.
4. **ESCO or O\*NET taxonomy** — replaces the hand-maintained alias map at scale.
5. **~1,000 pairs and a fine-tuned reranker** — raises the ceiling instead of
   approaching it.
