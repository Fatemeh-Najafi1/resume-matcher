# Evaluation protocol

How this project decides whether a change helped, and what the resulting
numbers may and may not be used to claim.

The short version: **one set to tune on, two sets to be measured by, and a
rule that the second kind never informs a decision.** Everything below is the
consequence of taking that seriously.

---

## Why this document exists

An earlier version of this project reported precision 0.91 / recall 0.82 and
attached a disclaimer: the numbers were contaminated, because the skill alias
map and the skill vocabulary had both been repaired *after* inspecting which
pairs the matcher got wrong. That is textbook test-set leakage, and the
disclaimer was honest but useless — it told a reader the number was wrong
without telling them by how much, which is not meaningfully different from not
reporting it.

The fix for leakage is not a disclaimer. It is a second set that the repairs
never touched. This document defines that set and the procedure that keeps it
clean.

---

## The three sets

| Set | File | Pairs | Role |
|---|---|---|---|
| Development | `data/eval/pairs.jsonl` | 124 | Tuning. Thresholds, aliases, vocabulary and rules are all chosen here. Contaminated **by design** — that is what it is for. |
| Development, part 2 | `data/eval/pairs_multidomain.jsonl` | 55 | Also tuning. A second persona (full-stack / security) across web, mobile, security, testing and cloud. Added because one résumé is not a distribution. Deliberately **not** data engineering — that is the held-out set's domain, and widening dev toward it would fit the project to the held-out distribution sideways. |
| Held-out (English) | `data/eval/holdout.jsonl` | 60 | Measurement. Different persona, different domains. Never tuned against. |
| Bilingual (AR/EN) | `data/eval/pairs_bilingual.jsonl` | 44 | Measurement. Cross-language matching and Arabic orthography. Never tuned against. |

Each row is a `(requirement, resume bullet, label)` triple where `label = 1`
means the bullet is genuine evidence that the candidate meets the requirement.

### Why the held-out set uses a different persona

The development set is built around the ML resume in `resume.txt`. Reusing
that resume for the held-out set would have tested generalisation to new
*pairs* while leaving generalisation to new *vocabulary* untested — and the
alias map is precisely a piece of vocabulary that was fitted to the
development set.

So the held-out set is built from a mid-level **data engineer**: Airflow, dbt,
Snowflake, Terraform, CDC, Parquet. None of those terms drove any tuning
decision. This turned out to be the single most informative choice in the
whole evaluation — see *What the held-out set revealed* below.

---

## The protocol

1. **Thresholds and calibration are fitted on the development set only.**
   `eval/holdout.py` refits them on every run rather than reading them from
   `config.yaml`, so the held-out number can never quietly inherit a setting
   that was tuned elsewhere.
2. **Held-out sets are run after a change is finalised, not during.** They are
   a measurement, not a signal.
3. **No setting may be chosen by looking at a held-out number.** Not a
   threshold, not an alias, not a vocabulary entry, not a rule.
4. **If rule 3 is ever broken, the set is burned** and must be replaced with
   newly written pairs. A held-out set that has influenced a decision is a
   development set wearing the wrong label.

This is enforced by procedure, not by code. That is a real weakness and it is
worth naming: nothing stops a future commit from tuning against
`holdout.jsonl`. The mitigation is that the development set exists, is
comfortable to use, and is the path of least resistance.

---

## What is measured

### Classification quality

Precision, recall and F1 for the positive class, with **precision as a
constraint rather than an objective**. Thresholds are chosen as *the
highest-recall point at which precision stays at or above 0.90*, not the point
that maximises F1.

This is not a neutral choice and it should not be read as one. A false
positive here tells someone they meet a requirement they do not — the exact
failure the project exists to prevent, and the one that costs them an
interview rather than an application. Recall is bought only with precision the
constraint can spare.

### Selective prediction

The matcher is allowed to abstain: it withholds a verdict on personal
qualities, and it reports `unsupported` rather than guessing when the resume is
silent. Scoring it as if it always answered measures a system that does not
exist.

So `eval/holdout.py` also reports, via `eval/metrics.py`:

- **Risk–coverage curve** — the error rate when answering only the *n* most
  confident decisions, from 30% coverage to 100%. This supports the claim
  worth making: not "precision 0.91", but "answering the 70% it is surest of,
  it is wrong *x*% of the time".
- **AURC**, with a **shuffled-confidence baseline** beside it. AURC alone is
  unreadable; the gap between it and the baseline is the part attributable to
  the confidence signal rather than to the underlying accuracy.
- **Abstain ECE** — whether a stated confidence of 0.8 corresponds to being
  right 80% of the time. A system can rank its answers well and still be
  systematically overconfident about all of them, which matters because the
  confidence is shown to a user.
- **Coverage at a risk ceiling** — "to stay under 5% error, it can answer this
  fraction of requirements."

Ties in confidence are averaged rather than ordered. The first implementation
sorted them by input order, which made AURC depend on the order of the data
file and scored a completely uninformative confidence signal at 0.04. The test
that caught it is `tests/test_metrics.py::test_aurc_beats_chance_only_when_confidence_informs`.

### Cost

Every scorer is timed on the same pairs, and its on-disk footprint is measured
from the local model cache rather than quoted. Accuracy tables without a cost
column hide the question that actually decides a deployment: whether two
points of F1 are worth forty times the latency on a 1GB free tier.

### The baseline ladder

Six rungs, each measured identically, so the contribution of every layer is
visible rather than assumed:

| Rung | What it is | Model |
|---|---|---|
| Keyword overlap | Jaccard over content words | none |
| BM25 | Lexical ranking with IDF weighting | none |
| Skills mapping | The literal gate alone — vocabulary, aliases, degree and duration rules | none |
| Semantic search | Bi-encoder cosine | ~90MB |
| Ranking | Cross-encoder over the bi-encoder's shortlist | +~90MB |
| Entailment | NLI: does the bullet *mean* the candidate has this? | +~90MB |

The bottom rung matters most. Anything that cannot beat Jaccard overlap has no
business being in the pipeline, and reporting it keeps the model rungs honest
about how much they actually add.

---

## What the held-out set revealed

Running the held-out set for the first time produced the most useful result in
the project's history, and it was a negative one.

**Synonym handling collapsed.** On the original 124-pair development set the
matcher scored 8/8 on the synonym category. On the held-out set it scored 2/7.

The cause is exactly the contamination the old README admitted to but could
not quantify. The alias map was built by reading development-set failures, so
it contains the synonyms that set happens to use — Git for version control,
XGBoost for gradient boosting. It knows nothing about Terraform meaning
infrastructure-as-code, Parquet meaning columnar storage, or Airflow meaning
workflow orchestration, because no development-set pair ever asked it to.

Then the development set was widened to a second persona, and the numbers
settled into something more honest:

| Category | Dev (124, one persona) | Dev (179, two personas) | Held-out |
|---|---|---|---|
| synonym | 8/8 | 12/21 | 2/7 |
| paraphrase | 12/14 | 14/23 | 2/7 |
| F1 (shipped scorer) | 0.86 | 0.76 | 0.62 |

Read the middle column. **Widening the development set dropped its own F1 from
0.86 to 0.76 without a line of matcher code changing.** Ten of those points
were never accuracy; they were fit. And the synonym score fell from a perfect
8/8 to 12/21 — the weakness is now visible on the surface decisions are
allowed to be made on, which is the whole point of having one.

That is the strongest argument here for the cost of a wider development set. A
held-out set tells you that you have a problem; it cannot be used to fix it
without being destroyed. A development set that reflects the real distribution
tells you the same thing *and* lets you work on it.

Three conclusions follow:

1. The original development-set synonym score was never a measurement of
   synonym handling. It was a measurement of how completely the alias map had
   been fitted to one resume.
2. **A hand-maintained alias map does not generalise**, and a narrow
   development-set score actively conceals that.
3. The fix is not more aliases. Adding data-engineering terms would simply
   re-fit the map to the held-out set and burn it. The fix is a vocabulary
   that was never hand-fitted at all — which was tried, and did not work
   either. See *What did not work*.

---

## What did not work

The held-out run gave a clear diagnosis, so the obvious next step was to act on
it. Six changes were tried. **None of them beat the baseline**, and the reasons
are more useful than a win would have been.

First, the measurement that framed everything. On the 179-pair development set,
split by which half of the system made each call:

| Path | Decisions | Accuracy |
|---|---|---|
| Literal gate (deterministic) | 82 (46%) | **93.9%** |
| Cosine similarity | 97 (54%) | **62.9%** |

The deterministic path is thirty-one points more accurate and sees fewer than
half the decisions. That single table reframed the problem: the lever is
**coverage of the deterministic path**, not the quality of the model behind the
other one. Every attempt below follows from it.

### 1. Fusing lexical and dense scoring — rejected before building

Hypothesis: BM25 and the bi-encoder miss different pairs, so combining them
recovers recall.

Measured on dev first: BM25 wrong alone on 3 pairs, the bi-encoder wrong alone
on 5, **both wrong on 12**. The errors are correlated, not complementary, so
fusion had little to recover. Not built.

### 2. A much better encoder — no effect

Hypothesis: `all-MiniLM-L6-v2` is far behind current models on public
benchmarks, so upgrading it should fix the paraphrase and synonym failures.

`bge-large-en-v1.5` is roughly fourteen times the size and well ahead on those
benchmarks:

| Encoder | F1 | synonym | paraphrase |
|---|---|---|---|
| all-MiniLM-L6-v2 (shipped) | 0.82 | 8/8 | 11/14 |
| bge-large-en-v1.5 | 0.83 | 8/8 | 11/14 |

One point of F1 and **not a single additional synonym or paraphrase case**. The
encoder is not the bottleneck, and the public benchmark ranking did not transfer
to this task at all. This is the result that killed "just use a better model" as
a strategy.

### 3. ESCO as an external vocabulary — wrong source

ESCO is the EU's occupational taxonomy, CC BY 4.0, and this project's own notes
had named it the top fix. Checked against its API before any integration work:

| Query | Hits |
|---|---|
| python | 2 |
| javascript | 6 |
| kubernetes | **0** |
| docker | **0** |
| terraform | **0** |
| machine learning | 894 (fuzzy prose matches, not skill entries) |

ESCO describes occupations and transversal skills well and named technologies
not at all. The recommendation to use it had been made without checking it;
checking it took ten minutes and saved several days.

### 4. A Stack Overflow vocabulary — measurably worse

If a hand-written vocabulary can only contain what its author thought of, the
fix is a vocabulary from outside the project. Stack Overflow's tag list is
exactly that: a technology vocabulary maintained by millions of practitioners,
with a synonym graph (`js` → `javascript`, `rails` → `ruby-on-rails`) that is
the same relation this project encodes by hand. `tools/fetch_vocabulary.py`
pulls it, filters out ordinary English words using WordNet, and writes 779
terms against the hand-written 130.

It recognised Airflow, Snowflake, Docker, Kubernetes, React and TypeScript —
none of which the hand-written list contained, and two of which are held-out
domain terms that arrived from outside rather than from inspecting failures.

And it lost:

| Configuration | Gate decisions | Gate accuracy | F1 |
|---|---|---|---|
| Hand-written vocabulary | 82 | **93.9%** | **0.70** |
| + generated vocabulary | 84 | 89.3% | 0.68 |

### 5. Blaming the generated aliases — wrong suspect

The natural reading of #4 was that 361 new alias entries had introduced noise.
An ablation disagreed, and it carried a built-in correctness check: its third
row reproduces the baseline exactly, so the harness was measuring what it
claimed to.

| Configuration | Gate | Gate accuracy | F1 |
|---|---|---|---|
| Generated vocabulary + aliases | 84 | 89.3% | 0.68 |
| Generated vocabulary, aliases OFF | 84 | 89.3% | 0.68 |
| Both OFF | 82 | 93.9% | 0.70 |

Turning the aliases off changed **nothing**. The vocabulary was the culprit.

Which was puzzling, because only **6 of 728** new terms fire on requirement
text at all. Those six: `apache spark`, `databricks`, `database design`,
`responsive design`, `backend`, `frontend`. Four of them are **concepts, not
product names** — and that is the whole story. "3+ years of frontend
experience" made the gate hunt for the literal string "frontend" in a bullet
reading *"React, TypeScript, Tailwind"*, find nothing, and veto a genuine match.

### 6. Separating products from concepts — principled, and still worse

The fix implied by #5: a named technology's absence is informative (PyTorch is
either written or not), a concept's absence is not (people do front-end work
for a decade without typing the word). So let only specific technologies drive
a veto, and defer the rest to similarity. Specificity was decided by WordNet —
a term is specific if it carries a token no dictionary knows.

| Configuration | Gate | Gate accuracy | F1 |
|---|---|---|---|
| Baseline | 82 | 93.9% | **0.70** |
| Concept split, generated vocabulary on | 56 | 92.9% | 0.66 |
| …plus an exception for concepts with aliases | 66 | 90.9% | 0.66 |
| Concept split, hand-written vocabulary | 65 | 93.8% | 0.68 |

It did exactly what it was designed to do — gate accuracy recovered to 93.8% —
and still lost, because it bought that accuracy by **giving up 17 decisions to
a path that is thirty points worse**. Accuracy on the decisions you keep is not
the objective when abstention hands the work to something weaker.

Reverted. The helper was deleted rather than left in as dead code.

### What this leaves

The generated vocabulary is kept, documented, and **off by default**
(`skills.use_generated` in `config.yaml`), so the comparison is reproducible
without shipping a regression.

One caveat has to be stated plainly, because it undercuts all six results: **the
development set is also where the hand-written vocabulary was fitted.** Every
comparison above is between an incumbent tuned on this data and a challenger
that was not, scored on that same data. Widening the set to 179 pairs across two
personas mitigates it; it does not remove it. These results are strong enough to
justify not shipping the changes, and not strong enough to conclude the
hand-written vocabulary is genuinely better. Settling that needs a fresh set, and
spending the held-out set on it would destroy the one instrument that works.

The honest summary is narrower than "nothing helps":

- The bottleneck is **coverage of the deterministic path**, measured.
- It is **not** the encoder, measured twice.
- It is **not** solved by a larger vocabulary, because requirements are phrased
  as concepts while external vocabularies list products.
- The missing piece is the **concept → instance** relation ("container
  orchestration" → Kubernetes), which tag synonyms do not provide, ESCO does
  not cover, and this project currently supplies by hand — one concept at a
  time, fitted to whatever failed last.

That is a sharper open problem than the one this started with, and it is the
next thing worth building.

## What these numbers may be used to claim

**Can be claimed:**
- The *ranking* of scorers against each other on this data.
- The held-out figures, as an estimate on data of this kind, with the
  distribution caveat below.
- That the matcher's confidence signal is informative, if AURC beats its
  shuffled baseline.

**Cannot be claimed:**
- That these figures predict performance on real recruiting traffic. All three
  sets are curated collections of hard cases — topically-adjacent negatives,
  paraphrases, degree levels — deliberately over-weighting what the matcher
  must not get wrong. Real postings contain many easy pairs all three sets
  omit, so these numbers are *pessimistic* about the average case and say
  nothing about the tail.
- Anything about fairness across demographic groups. Not measured here.
- That the development-set figures mean anything at all as absolutes. They are
  an upper bound and are reported only to show the size of the gap.

**A note on who wrote the labels.** All three sets were labelled by one person,
so the inter-annotator agreement rate — the ceiling no model can beat — is
unknown. On the `generic` category the development set caught its own author
being inconsistent: mentoring was marked as evidence of "team leadership" but
not of "works in a fast-paced environment". That inconsistency is why the app
now abstains on personal qualities instead of guessing. A second annotator on
the existing pairs would establish the ceiling and remains an open item.

---

## Reproducing

```bash
pip install -r requirements.txt

python -m pytest tests/ -q          # the claims this README makes, as tests
python -m eval.run_eval             # development set: tune, compare, inspect errors
python -m eval.run_eval --errors    # ...and print what it got wrong
python -m eval.holdout              # held-out English set
python -m eval.holdout --set bilingual
python -m eval.holdout --all        # every set, side by side
python -m eval.holdout --quick      # skip the two heavyweight models
```

First run downloads models (~90MB each) and the NLTK corpora. Everything is
CPU-only and deterministic: `temperature=0` wherever a model is sampled, fixed
seeds in the metrics, and no network dependency once the models are cached.

---

## Open items

1. **A source for the concept → instance relation.** This is the real open
   problem, and it is sharper than the one this started with. Requirements are
   written as concepts ("container orchestration", "infrastructure as code");
   vocabularies list products (Kubernetes, Terraform). Tag synonyms give
   spelling variants, not hypernyms. ESCO does not cover technology at all —
   both checked and recorded under *What did not work*. Candidates not yet
   tried: Wikipedia/Wikidata category graphs, DBpedia, or deriving the
   relation from co-occurrence in a large job-posting corpus.
2. **A second annotator** on the existing pairs, to establish the agreement
   ceiling.
3. **Per-domain reporting at scale.** A retrieval benchmark in this area found
   variance across industry domains exceeded the gains from any module
   upgrade, which means a single headline number is misleading by
   construction.
4. **A fairness audit** — perturb names and demographic markers on a fixed set
   and report score deltas. The deterministic rungs should show near-exact
   invariance, which is a claim no LLM scorer can make. Not yet run.
5. **More bilingual pairs.** 44 is enough to detect a gross failure and not
   enough to compare models with confidence.
