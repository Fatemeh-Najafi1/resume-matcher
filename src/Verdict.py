# src/Verdict.py
"""The verdict vocabulary, and the rule that separates silence from mismatch.

Every matching tool this project was compared against collapses "not covered"
into a single bucket. That bucket hides the distinction a recruiter actually
needs, because two very different things land in it:

  * The resume says nothing either way. The candidate may well hold the skill
    and simply not have written it down. This is MISSING INFORMATION, and the
    correct response is to ask.
  * The resume states something incompatible with the requirement — a
    bachelor's against a posting demanding a PhD, two years against a demand
    for eight. This is GENUINE UNSUITABILITY, and the correct response is to
    stop.

Conflating them is how a qualified candidate gets filtered out for an omission
rather than for a shortfall. So this module keeps them apart:

    covered       the resume evidences the requirement
    partial       some of the requirement is evidenced, some is not
    unsupported   the resume is silent; this is a question, not a rejection
    contradicted  the resume states something that cannot satisfy the demand
    judgement     a personal quality a resume cannot settle either way

Two asymmetries are deliberate and worth stating, because they are what keep
the `contradicted` bucket honest:

1. **Absence is never contradiction.** A resume that never mentions Kafka is
   `unsupported` on Kafka, not `contradicted`. Nothing about an unwritten line
   proves an unheld skill. Only a stated, checkable value can contradict.

2. **Contradiction needs a stated value on both sides.** The requirement must
   name a floor (a degree level, a number of years) and the resume must state
   its own value. If either side is silent, the verdict falls back to
   `unsupported` or `partial`.

Both asymmetries push errors toward "ask the candidate" and away from "reject
the candidate", which is the direction this project errs in everywhere else.
"""

import re

from .utils import degree_level, degree_required

COVERED = "covered"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
CONTRADICTED = "contradicted"
JUDGEMENT = "judgement"

#: Every verdict the matcher can return, in the order a reader should scan
#: them: the blocking ones first, the settled ones last.
ORDER = (CONTRADICTED, UNSUPPORTED, PARTIAL, COVERED, JUDGEMENT)

#: What each verdict means, in the words the UI should use.
MEANING = {
    COVERED: "Your resume evidences this.",
    PARTIAL: "Partly evidenced — some of what this asks for is shown, some is not.",
    UNSUPPORTED: "Your resume does not speak to this. It may still be true — "
                 "consider adding it if you have it.",
    CONTRADICTED: "Your resume states something that does not meet this demand.",
    JUDGEMENT: "A personal quality a resume cannot settle — your call.",
}

#: Verdicts that count as a genuine mismatch rather than an open question.
#: Only this set should ever be read as "this candidate does not qualify".
BLOCKING = frozenset({CONTRADICTED})

_DEGREE_NAMES = {0: "no degree", 1: "a bachelor's", 2: "a master's", 3: "a PhD"}

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# "5+ years", "at least three years", "minimum of 2 yrs", and the Arabic
# equivalents ("٥ سنوات", "خمس سنوات"). Arabic-Indic digits are normalised
# before this runs, so only the Latin forms need matching here.
_YEARS = re.compile(
    r"(\d+|" + "|".join(_WORD_NUMBERS) + r")\s*\+?\s*(?:years?|yrs?|سنوات|سنة|عام|أعوام)",
    re.IGNORECASE,
)

# A resume stating its own total: "3 years of experience", "over 5 years in".
# Deliberately narrower than _YEARS, which also fires on "2023-2025".
_YEARS_STATED = re.compile(
    r"(?:over|more than|nearly|about|approximately|\+)?\s*"
    r"(\d+|" + "|".join(_WORD_NUMBERS) + r")\s*\+?\s*"
    r"(?:years?|yrs?|سنوات|سنة|عام|أعوام)\s*"
    r"(?:\+)?\s*(?:of|in|as|with|experience|خبرة|من)",
    re.IGNORECASE,
)

#: How far a date-span estimate may fall short of a demand before the verdict
#: is allowed to become `contradicted`. Date spans are noisy — a degree start
#: year inflates them, a career gap deflates them — so a shortfall inside this
#: margin is reported as `partial` instead. Only an explicitly stated figure
#: contradicts without a margin.
DATE_SPAN_MARGIN_YEARS = 2.0


def _to_number(token):
    token = token.strip().lower()
    if token.isdigit():
        return float(token)
    return float(_WORD_NUMBERS.get(token, 0)) or None


def years_required(requirement_text):
    """The largest number of years the requirement demands, or None.

    Takes the largest rather than the first so that "3 to 5 years" is read as
    the demand it is, and a posting that mentions "2 years" in passing before
    asking for "8 years of production experience" is not under-read.
    """
    found = [_to_number(m.group(1)) for m in _YEARS.finditer(requirement_text)]
    found = [f for f in found if f]
    return max(found) if found else None


def years_claimed(resume_text):
    """How many years of experience the resume claims, and on what basis.

    Returns (years, basis) where basis is:
      * "stated" — the resume says so in words ("4 years of backend work").
        Trustworthy, so a shortfall against it contradicts outright.
      * "dates"  — inferred from the span between the earliest and latest year
        mentioned anywhere. Noisy: an education start date stretches it, so a
        shortfall against it only contradicts beyond DATE_SPAN_MARGIN_YEARS.
      * None     — the resume gives nothing to go on.
    """
    stated = [_to_number(m.group(1)) for m in _YEARS_STATED.finditer(resume_text)]
    stated = [s for s in stated if s]
    if stated:
        return max(stated), "stated"

    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", resume_text)]
    if len(years) >= 2:
        return float(max(years) - min(years)), "dates"

    return None, None


def resolve_degree(requirement_text, resume_text):
    """Verdict for a requirement naming a degree level, or None if it doesn't.

    The three-way split is the whole point. A posting demanding a master's
    against a resume that:
      * names a master's or better  -> covered
      * names a bachelor's          -> contradicted (a stated, insufficient level)
      * names no degree at all      -> unsupported  (silence, not a shortfall)

    That last line is the one every keyword tool gets wrong. A resume with no
    education section is not a resume without an education.
    """
    # The floor the posting will accept, not the ceiling it mentions:
    # "Master's or PhD" is met by a master's.
    needed = degree_required(requirement_text)
    if not needed:
        return None

    held = degree_level(resume_text)

    if held >= needed:
        return COVERED, f"your resume shows {_DEGREE_NAMES[held]}", []

    if held == 0:
        return (UNSUPPORTED,
                f"this asks for {_DEGREE_NAMES[needed]}; your resume does not state a "
                f"degree, so this is unanswered rather than unmet", [])

    return (CONTRADICTED,
            f"this asks for {_DEGREE_NAMES[needed]}; your resume states "
            f"{_DEGREE_NAMES[held]}", [])


def resolve_duration(requirement_text, resume_text, present, describe):
    """Verdict for a requirement demanding a length of experience, or None.

    Called only once the named skills are known to be present — the question
    here is no longer "do they know this?" but "for long enough?".

    Naming a tool never proves a duration, so the best case is `partial`. But
    when the resume states a figure of its own and that figure falls short, the
    shortfall is real and gets said plainly.
    """
    needed = years_required(requirement_text)
    if not needed:
        return None

    claimed, basis = years_claimed(resume_text)

    if claimed is not None and claimed >= needed:
        return (COVERED,
                f"found {describe(present)}, and your resume indicates about "
                f"{claimed:.0f} years against the {needed:.0f} asked for", present)

    if claimed is not None:
        shortfall_is_certain = (
            basis == "stated"
            or (needed - claimed) > DATE_SPAN_MARGIN_YEARS
        )
        if shortfall_is_certain:
            how = ("your resume states" if basis == "stated"
                   else "the dates on your resume span")
            return (CONTRADICTED,
                    f"this asks for {needed:.0f} years; {how} about "
                    f"{claimed:.0f}", present)

    return (PARTIAL,
            f"found {describe(present)}, but this asks for {needed:.0f} years of it, "
            f"which a skills list cannot prove", present)


def coverage(counts, decidable):
    """The headline coverage figure, and what it deliberately leaves out.

    `judgement` rows are excluded from the denominator: counting them either
    way would be a guess dressed up as a measurement. `unsupported` and
    `contradicted` both count as not-covered here — the distinction between
    them is for the reader, not for the arithmetic, because weighting one
    against the other would require knowing how likely an omission is to be
    real, which this tool cannot know.
    """
    if not decidable:
        return 0.0
    return (counts.get(COVERED, 0) + 0.5 * counts.get(PARTIAL, 0)) / decidable
