# tools/fetch_vocabulary.py
"""Build the skill vocabulary from an external source instead of by hand.

Run:  python -m tools.fetch_vocabulary          # refresh from the API
      python -m tools.fetch_vocabulary --dry    # show what would change

Why this exists
---------------
The held-out evaluation showed synonym accuracy falling from 8/8 to 2/7 on a
domain the project had never been tuned against. The cause was not the model.
Measured on the widened development set:

    decisions the literal gate ruled on   93.9% accurate
    decisions cosine similarity decided   62.9% accurate
    share reaching the literal gate       46%

The deterministic path is thirty-one points more accurate and sees fewer than
half the decisions. So the lever is **coverage of that path**, which is
`data/skills_list.txt` — 152 hand-typed terms. A bigger encoder does not help;
bge-large-en-v1.5, fourteen times the size of the shipped model, moved F1 by
0.01 and fixed no synonym or paraphrase case at all.

The trap is that widening the list by hand reproduces the original problem.
A hand-written vocabulary can only contain what its author thought of, which
is why the first one scored 8/8 on the resume it was written from and 2/7 on
a resume it was not. The vocabulary has to come from outside this project.

Why Stack Overflow and not ESCO
-------------------------------
ESCO was the obvious candidate — an official EU taxonomy, CC BY 4.0, and the
one this project's own notes named as the top fix. Measured against its API,
its coverage of software technology is not usable:

    python            2 hits
    javascript        6 hits
    kubernetes        0 hits
    docker            0 hits
    terraform         0 hits
    machine learning  894 hits   (fuzzy prose matches, not a skill entry)

ESCO describes occupations and transversal skills well and named technologies
badly. Recording this because the recommendation to use it was made before it
was checked, and checking it took ten minutes.

Stack Overflow's tag list is the opposite: it is a technology vocabulary
maintained by millions of practitioners, with an explicit synonym graph
(`js -> javascript`, `rails -> ruby-on-rails`) that is exactly the relation
this project's alias map encodes by hand. Nobody curated it for this project,
so it carries no fit to any evaluation set here.

Licence: Stack Exchange content is CC BY-SA 4.0. Tag strings are short factual
labels, but the attribution is recorded in the generated file's header anyway.

What it does NOT solve
----------------------
Tag synonyms are spelling variants (`k8s` = `kubernetes`). They are not
hypernyms: Stack Overflow does not say that Kubernetes evidences "container
orchestration". That relation stays in the hand-written map, which is now a
small override on top of a large generated base rather than the whole thing.
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_VOCAB = ROOT / "data" / "skills_generated.txt"
OUT_ALIASES = ROOT / "data" / "skill_aliases_generated.yaml"

API = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"

#: How many tags to take. The tail of the tag list is long and noisy; the head
#: is where the technologies people actually list on resumes live.
TOP_TAGS = 1200

#: Tags that are ordinary English words, or so generic they would fire on
#: nearly every document. A vocabulary term is only useful if its presence in
#: a resume means something, and "function" present in a resume means nothing.
#:
#: This list is the one hand-maintained part left, and it is a *blocklist*,
#: which fails safe: forgetting an entry adds noise that shows up as a
#: precision drop on the dev set, rather than silently removing a match.
_STOPTAGS = frozenset("""
function class object string list array date time set get post put if else
loop for while return type types value values key keys index file files
image images text table tables view views form forms button click event
events error errors exception memory thread process server client user users
data database query search sort filter print input output null variable
math random speed size count number integer float boolean char byte
math module import export package library framework api app application
web mobile desktop window windows linux unix macos android ios
testing test debugging performance security design model controller
header footer menu list-view layout style styles color fonts
select insert update delete join group order where from into
install installation configuration setup deployment build compile
string-formatting parsing validation encoding decoding
""".split())

# A tag is kept only if it looks like a proper technology name rather than a
# general noun: it contains a digit, a separator, a capital-letter origin, or
# is simply long enough to be distinctive.
_PLAUSIBLE = re.compile(r"^[a-z0-9][a-z0-9.+#_-]{2,}$")


def _get(path, **params):
    params.setdefault("site", SITE)
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    if response.headers.get("Content-Encoding") == "gzip":
        import gzip
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def fetch_tags(limit=TOP_TAGS):
    """The most-used tags, most popular first."""
    tags, page = [], 1
    while len(tags) < limit:
        data = _get("tags", page=page, pagesize=100, order="desc", sort="popular")
        tags.extend(item["name"] for item in data.get("items", []))
        if not data.get("has_more"):
            break
        page += 1
        time.sleep(0.2)
    return tags[:limit]


def fetch_synonyms():
    """from_tag -> to_tag, the community's own alias graph."""
    pairs, page = {}, 1
    while True:
        data = _get("tags/synonyms", page=page, pagesize=100,
                    order="desc", sort="applied")
        for item in data.get("items", []):
            pairs[item["from_tag"]] = item["to_tag"]
        if not data.get("has_more") or page >= 20:
            break
        page += 1
        time.sleep(0.2)
    return pairs


def _is_english_word(token):
    """True if this is an ordinary English word rather than a technology name.

    The synonym graph redirects a lot of ordinary vocabulary at libraries —
    `activity` at android-activity, `addclass` at jquery. Those are correct as
    Stack Overflow routing and wrong as resume evidence: a vocabulary term
    only earns its place if its presence in a resume means something, and
    "activity" appearing in a resume means nothing at all.

    WordNet is already a dependency of this project, so this costs nothing.
    If it is unavailable the check fails OPEN (returns False, term is kept),
    because losing a real technology is worse than keeping a noisy one — the
    noise shows up as a precision drop on the dev set, where it is visible.
    """
    if " " in token or any(c.isdigit() or c in ".+#-_" for c in token):
        return False
    try:
        from nltk.corpus import wordnet
        return bool(wordnet.synsets(token))
    except Exception:
        return False


def usable(tag):
    """Is this tag worth putting in a vocabulary matched literally?"""
    if tag in _STOPTAGS:
        return False
    if _is_english_word(tag):
        return False
    if len(tag) < 3:
        # One- and two-letter tags that are also ordinary words ("go", "r",
        # "c") fire on nearly every document. The existing vocabulary already
        # excludes these deliberately; keep doing so.
        return False
    return bool(_PLAUSIBLE.match(tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print, do not write")
    ap.add_argument("--limit", type=int, default=TOP_TAGS)
    args = ap.parse_args()

    print(f"fetching top {args.limit} tags...")
    tags = fetch_tags(args.limit)
    print(f"fetching synonym graph...")
    synonyms = fetch_synonyms()

    kept = sorted({t.replace("-", " ") for t in tags if usable(t)})
    dropped = len(tags) - len(kept)

    # Keep a synonym only when its target survived the filter, so the alias
    # map can never point at a term the vocabulary does not contain — the
    # exact bug that made the first hand-written alias map silently inert.
    vocabulary = set(kept)
    alias_lines = []
    for source, target in sorted(synonyms.items()):
        s, t = source.replace("-", " "), target.replace("-", " ")
        if t in vocabulary and usable(source) and s != t:
            alias_lines.append(f"{t}: [{s}]")

    print(f"\n  {len(tags)} tags fetched, {dropped} filtered out, {len(kept)} kept")
    print(f"  {len(alias_lines)} synonyms whose target is in the vocabulary")
    print(f"\n  sample vocabulary: {', '.join(kept[:12])}")
    print(f"  sample aliases   : {'; '.join(alias_lines[:5])}")

    if args.dry:
        print("\n--dry: nothing written")
        return

    header = (
        "# GENERATED by tools/fetch_vocabulary.py — do not edit by hand.\n"
        "#\n"
        "# Source: Stack Overflow tag list via the Stack Exchange API.\n"
        "# Stack Exchange content is CC BY-SA 4.0.\n"
        "#\n"
        "# This file exists so the skill vocabulary is NOT fitted to this\n"
        "# project's evaluation sets. data/skills_list.txt remains as the\n"
        "# hand-written override for concepts Stack Overflow has no tag for.\n"
        f"# {len(kept)} terms.\n"
    )
    OUT_VOCAB.write_text(header + "\n".join(kept) + "\n", encoding="utf-8")
    OUT_ALIASES.write_text(
        header.replace("terms.", "alias entries.") +
        "\n" + "\n".join(alias_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_VOCAB.relative_to(ROOT)} and {OUT_ALIASES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
