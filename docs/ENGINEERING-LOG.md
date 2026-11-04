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

> **Resolved — see 11.** The held-out set was built and run. The
> contamination was real and larger than expected: F1 0.86 on the tuned
> set, 0.65 on the untouched one. The claim above that the *ranking* was
> sound also turned out to be wrong — see 12.


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

## 11. The held-out set, and what it cost me

I built `data/eval/holdout.jsonl` — 60 pairs from a **different persona** (a
mid-level data engineer: Airflow, dbt, Snowflake, Terraform, CDC, Parquet) and
ran it once. Choosing a different persona rather than more pairs from the same
resume was the single most informative decision in the project, and I nearly
did not make it.

Reusing `resume.txt` would have tested generalisation to new *pairs* while
leaving generalisation to new *vocabulary* untested — and the alias map is
precisely a piece of vocabulary that was fitted to the dev set. The test would
have passed and told me nothing.

**The result:**

| | Dev | Held-out |
|---|---|---|
| F1 (cross-encoder + gate) | 0.86 | 0.65 |
| Synonym category | 8/8 | 2/7 |
| Paraphrase category | 12/14 | 3/7 |
| Abstain ECE | 0.005 | 0.178 |

§8 said the absolute values were unconfirmed but the ranking was sound. Half
of that was right. The absolute values were indeed inflated — by 21 points of
F1. But the ranking was **not** sound, which is §12.

The synonym collapse has a precise cause. The alias map holds the synonyms the
dev set happens to use: Git for version control, XGBoost for gradient
boosting. It has never heard of Terraform meaning infrastructure-as-code. The
dev-set synonym score was never measuring synonym handling at all — it was
measuring how completely I had fitted the map to one resume.

The tempting fix is to add the data-engineering terms. That would re-fit the
map to the held-out set and burn it, and six months later I would be writing
this section again about a third set. The actual fix is a vocabulary nobody
hand-fitted, which is why ESCO alternative labels are now the top open item.

---

## 12. The cross-encoder ranking reversed. Again.

§6 records the ranking flipping once already, when pairs 91–124 added
junior-vs-senior cases. It has now flipped a second time, in the other
direction:

| | Dev F1 | Held-out F1 | ms/pair |
|---|---|---|---|
| Semantic search (bi-encoder) | 0.82 | **0.67** | 29 |
| Ranking (cross-encoder) | **0.86** | 0.65 | 182 |

Six times the latency to lose. Two reversals on two distribution changes is no
longer a fact about either model — it is a fact about the benchmark, which is
too small and too curated to separate them.

So I did not change the shipped default. That felt wrong for about a day. The
reasoning that settled it: flipping the default on the strength of a held-out
number would convert the held-out set into a development set, and I would have
traded a durable measurement instrument for two points of F1 I do not trust.
The question needs a third set. Until then the README says the question is
open, which is the true state of it.

This is the thing I would most want to be asked about.

---

## 13. Separating "not mentioned" from "ruled out"

The matcher used to report one bucket, `missing`, which held two unrelated
things: a resume that is silent about Kafka, and a resume that states a
bachelor's against a posting demanding a PhD. The first is an omission and the
second is a shortfall, and treating them alike is how a qualified person gets
filtered out for not having written something down.

`src/Verdict.py` splits them. Two asymmetries keep the split honest:

- **Absence is never contradiction.** Nothing about an unwritten line proves
  an unheld skill.
- **Contradiction needs a stated value on both sides.** A posting demanding a
  master's against a resume with *no* education section is `unsupported`, not
  `contradicted` — a resume without an education section is not a candidate
  without an education.

Both asymmetries push errors toward "ask the candidate" rather than "reject
the candidate", which is the direction this project errs in everywhere else.

Writing the tests for it surfaced a bug that had been there from the
beginning, in the original code, unrelated to the refactor: `degree_level`
returned the **highest** level named, and requirements were read with the same
function. So "Master's degree required, PhD preferred" was read as a demand
for a PhD, and every qualified master's-holder was told they fell short of a
posting that would have hired them. Requirements state a *floor*; resumes
state a *ceiling*. They needed different functions, and now have them.

---

## 14. Arabic broke four things, all of them silently

Adding Arabic was supposed to be a model swap. It was not.

1. **The English encoder does not fail on Arabic — it returns noise.**
   `all-MiniLM-L6-v2` has no Arabic in its training data. Handed an Arabic
   requirement it produces a vector, the matcher compares that vector to a
   resume, and the result is reported with exactly the confidence everything
   else gets. Final measurement: F1 0.70 against 0.78 for a multilingual
   encoder. The first measurement, taken before fixes 2-4 below, showed a far
   wider gap (0.40 against 0.62) because the literal gate was not firing in
   Arabic at all then and the encoder was carrying the whole load alone.
2. **`(?<![a-z0-9])` does nothing in Arabic.** An Arabic letter is not in
   `[a-z0-9]`, so the lookaround passed and vocabulary terms matched happily
   inside longer Arabic words.
3. **The degree marker list held no Arabic**, so `degree_required` returned 0
   for every Arabic requirement, the gate never fired, and cosine confidently
   returned "covered" for a master's demand against a resume with none.
4. **The alias keys were lowercased, not normalised.** `.lower()` is a no-op
   on Arabic, so the YAML key kept its madda alef while `find_skills` returned
   the folded form. Every Arabic alias lookup missed, and the gate reported
   "not found in your resume" about a skill that was sitting right there.

Every one of those is the same failure shape as the original truncation bug in
§2: no exception, no warning, a plausible number. I have now hit it five times
in this codebase, always at a boundary where one component assumes something
about text that another component does not guarantee.

The chunker was the fifth. It hardcoded 150 words for a 256-token window; the
multilingual model's window is 128. It would have silently truncated every
chunk and the Arabic results would have been quietly meaningless. It now
derives the stride from the loaded model.

**An honesty note I have to record against myself.** Fixes 2–4 were made after
the bilingual set had been run, which means the post-fix bilingual numbers are
not a clean held-out measurement for the affected categories. I made them
anyway because each was justified by *reading the code* — a marker list
containing no Arabic cannot match Arabic, which is not a benchmark finding —
but the set is partly spent and the README says so. The clean version needs
fresh pairs.

---

## 15. A metric bug the tests caught before it reached the README

The first risk-coverage implementation sorted by confidence and took the input
order for ties. The literal gate hands many decisions identical confidence, so
the curve — and the AURC computed from it — depended on the order of rows in
the data file.

The test that caught it feeds a deliberately uninformative confidence signal
and asserts the AURC is no better than chance. It scored 0.04, because the
correct rows happened to be listed first. Ties are now averaged within each
block, which is the expected curve over random tie-breaking computed
deterministically.

I would not have found this by inspection. It produced a *better* number, in a
metric nobody can sanity-check by eye, in the direction I wanted.

---

## Open items

1. **ESCO alternative labels** (CC BY 4.0) to replace the hand-maintained
   alias map, keeping the hand map as an override. This is the direct fix for
   the synonym collapse in §11. Published ESCO coverage reaches about 64.5% of
   annotated skill spans, so the hybrid is necessary rather than lazy.
2. **A third evaluation set** to settle the bi-encoder/cross-encoder question
   in §12. Neither existing set can.
3. **A second annotator** on the existing pairs. The disagreement rate is the
   ceiling no model can beat, and it is currently unknown.
4. **Fresh bilingual pairs**, to re-measure Arabic cleanly after the fixes in
   §14. 44 pairs is enough to catch a gross failure and not enough to compare
   models.
5. **Counterfactual fairness audit** — perturb names and demographic markers,
   report score deltas and four-fifths impact ratios. The deterministic rungs
   should show near-exact invariance, which is a claim no LLM scorer can make.
6. **Per-domain reporting.** Published retrieval work in this area finds
   cross-industry variance exceeding the gains from any module upgrade, which
   makes a single headline number misleading by construction.
