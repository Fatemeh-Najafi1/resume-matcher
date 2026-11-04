# src/Fidelity.py
"""Check that a rewritten resume line still says only what the original said.

The README claims this project will not invent skills you don't have, and for
the gap report that claim is structural: the report is a set difference with no
model in the code path, so there is no mechanism by which it could fabricate.

The optional LLM panel is different. It rewrites bullets, and its instruction
not to invent anything is a sentence in a prompt. That is a promise, not a
guarantee, and prompts are not load-bearing. Published reviews of commercial
tools in this category describe exactly the failure this invites — one tool is
repeatedly accused of "creating metrics out of thin air", and the failure is
costly in a specific way: an invented number survives the resume screen and
then collapses in the interview, which is the worst place to find out.

So every rewrite is checked after generation against the text it came from.
Three things cannot appear in a rewrite unless they appear in the source:

  * **a skill** that the resume never mentions — a fabricated capability;
  * **a number** the resume never states — the invented metric, the
    "improved throughput by 40%" nobody measured;
  * **a seniority claim** the original did not make — "led", "owned",
    "architected" grafted onto a line that described participation.

The first two are decidable by inspection and are rejected outright. The third
is a lexical proxy for something genuinely unsolved: a rewrite can imply
seniority without using any of these words, and no checker here will catch
that. It is reported as a warning rather than a rejection, and the limit is
stated in the README rather than hidden.

The asymmetry throughout is deliberate. A rewrite wrongly rejected costs the
user a suggestion. A rewrite wrongly accepted costs them an interview.
"""

import re

from .utils import find_skills, normalise

#: Verbs that assert a level of responsibility rather than describe work. A
#: rewrite may use one only if the original did.
_SENIORITY_CLAIMS = (
    "led", "leading", "managed", "managing", "owned", "owning", "architected",
    "founded", "headed", "directed", "supervised", "mentored", "spearheaded",
    "oversaw", "overseeing", "established", "built the team", "hired",
)

#: Numbers that carry no claim. "24/7" and version numbers are not metrics, and
#: flagging them produces noise that trains the reader to ignore the warnings.
_NOT_A_METRIC = re.compile(r"^(?:0|1|2|3|4|5|6|7|8|9|10|24|7|100)$")

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?%?")


def _numbers(text):
    """Quantities a reader would take as a claim."""
    out = set()
    for raw in _NUMBER.findall(text):
        token = raw.rstrip("%").replace(",", "")
        if raw.endswith("%") or not _NOT_A_METRIC.match(token):
            out.add(raw.rstrip("."))
    return out


def _seniority_claims(text):
    folded = normalise(text)
    return {verb for verb in _SENIORITY_CLAIMS
            if re.search(r"(?<!\w)" + re.escape(verb) + r"(?!\w)", folded)}


def check_rewrite(original, rewritten, resume_text=""):
    """Compare a rewritten bullet against its source.

    `resume_text` widens the evidence base: a rewrite may legitimately pull a
    skill from elsewhere in the same resume, which is a moved claim rather than
    an invented one. Passing it empty makes the check strictly bullet-local.

    Returns a dict:
        ok               False if anything was invented outright
        invented_skills  skills in the rewrite that appear nowhere in the source
        invented_numbers numbers in the rewrite that appear nowhere in the source
        moved_skills     skills found elsewhere in the resume but not in this
                         bullet — allowed, but worth the user's eye
        added_seniority  seniority verbs the original did not use
        problems         human-readable lines explaining each finding
    """
    source_skills = set(find_skills(original)) | set(find_skills(resume_text))
    bullet_skills = set(find_skills(original))
    new_skills = set(find_skills(rewritten))

    invented_skills = sorted(new_skills - source_skills)
    moved_skills = sorted((new_skills - bullet_skills) & source_skills)

    source_numbers = _numbers(original) | _numbers(resume_text)
    invented_numbers = sorted(_numbers(rewritten) - source_numbers)

    added_seniority = sorted(_seniority_claims(rewritten) - _seniority_claims(original))

    problems = []
    for skill in invented_skills:
        problems.append(
            f"claims **{skill}**, which appears nowhere in your resume")
    for number in invented_numbers:
        problems.append(
            f"introduces the figure **{number}**, which your resume never states")
    for verb in added_seniority:
        problems.append(
            f"adds **\"{verb}\"** — your original line did not claim that role")
    for skill in moved_skills:
        problems.append(
            f"mentions **{skill}** from elsewhere in your resume; check it belongs here")

    return {
        "ok": not invented_skills and not invented_numbers,
        "invented_skills": invented_skills,
        "invented_numbers": invented_numbers,
        "moved_skills": moved_skills,
        "added_seniority": added_seniority,
        "problems": problems,
    }


def screen_rewrites(rewrites, resume_text=""):
    """Run check_rewrite over the LLM's suggestions and attach the verdicts.

    Nothing is silently dropped. A rejected rewrite is still shown, marked and
    explained, because the useful information for the user is not "here are
    three suggestions" but "this tool tried to put a number in your mouth".
    Hiding the failure would make the panel look better and the user worse off.
    """
    screened = []
    for item in rewrites or []:
        if not isinstance(item, dict):
            continue
        original = (item.get("original") or "").strip()
        rewritten = (item.get("rewritten") or "").strip()
        if not original or not rewritten:
            continue
        screened.append({**item, "fidelity": check_rewrite(original, rewritten, resume_text)})
    return screened


def summarise(screened):
    """One line for the panel header: how many suggestions survived the check."""
    total = len(screened)
    failed = sum(1 for s in screened if not s["fidelity"]["ok"])
    if not total:
        return ""
    if not failed:
        return f"All {total} rewrites check out — none adds a claim your resume doesn't make."
    return (f"{failed} of {total} rewrites introduced something your resume never said. "
            f"They are shown below, marked.")
