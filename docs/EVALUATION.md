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

**Synonym handling collapsed.** On the development set the matcher scores 8/8
on the synonym category. On the held-out set it scored 2/7, and paraphrase fell from 12/14 to 3/7.

The cause is exactly the contamination the old README admitted to but could
not quantify. The alias map was built by reading development-set failures, so
it contains the synonyms that set happens to use — Git for version control,
XGBoost for gradient boosting. It knows nothing about Terraform meaning
infrastructure-as-code, Parquet meaning columnar storage, or Airflow meaning
workflow orchestration, because no development-set pair ever asked it to.

Two conclusions follow, and the second is the one worth keeping:

1. The development-set synonym score was never a measurement of synonym
   handling. It was a measurement of how completely the alias map had been
   fitted to one resume.
2. **A hand-maintained alias map does not generalise, and its development-set
   score actively conceals that.** The fix is not more aliases — adding
   data-engineering terms now would simply re-fit the map to the held-out set
   and burn it. The fix is a vocabulary that was never hand-fitted at all,
   which is why ESCO's alternative-label set is the top open item below.

This is what a held-out set is for. Nothing in months of development-set work
surfaced it.

---

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

1. **Replace the hand-maintained alias map with ESCO alternative labels**
   (CC BY 4.0), keeping the hand map only as an override. This is the direct
   fix for the synonym collapse. Published coverage of ESCO against annotated
   skill spans is about 64.5%, so a hybrid is necessary rather than lazy.
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
