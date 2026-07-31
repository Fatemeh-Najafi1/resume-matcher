# src/CompanyResearcher.py
import re
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# Words that mean we've matched prose, not a company name.
_NOT_A_COMPANY = {
    "the", "this", "our", "us", "we", "you", "your", "role", "team", "job",
    "position", "company", "employer", "opportunity", "candidate", "description",
}


def _looks_like_company(name: str) -> bool:
    """Reject the prose fragments the patterns below happily match."""
    if not (2 < len(name) <= 60):
        return False
    words = name.split()
    if not words or len(words) > 5:
        return False
    if all(w.lower() in _NOT_A_COMPANY for w in words):
        return False
    # A real name has letters; "US J.P" style fragments are mostly punctuation.
    letters = sum(c.isalpha() for c in name)
    return letters >= 3 and letters / len(name) > 0.5


def extract_company_name(job_text: str) -> str:
    """
    Best-effort company name from job description *text*.

    This is the fallback path. When the posting came from a URL, prefer the
    `company` field the scraper reads out of the page's structured data — that
    is what the site declares, rather than a guess from prose. Regexing prose
    is what turned a JPMorganChase posting into 'US\\nJ.P'.
    """
    # Patterns are matched against single lines, so a match can never run
    # across a line break and glue two unrelated fragments together.
    patterns = [
        r"\babout\s+(?:us\s+at\s+)?([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3})\b",
        r"\bjoin\s+(?:the\s+)?([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3})\s+team\b",
        r"\b([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3})\s+is\s+(?:a|an|the)\s+(?:leading|global|fast|growing|top)\b",
        r"\bcompany[:\s]+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3})\b",
        r"\bemployer[:\s]+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,3})\b",
    ]

    lines = [ln.strip() for ln in job_text.splitlines() if ln.strip()]
    for pattern in patterns:
        for line in lines:
            match = re.search(pattern, line)
            if match:
                name = match.group(1).strip().rstrip(".,;:")
                if _looks_like_company(name):
                    return name

    # LinkedIn postings put the employer on the second line, right after the
    # job title — the scraper writes it there.
    if len(lines) >= 2 and _looks_like_company(lines[1]):
        return lines[1]
    return ""


def research_company(company_name: str) -> dict:
    """
    Search for basic company info using DuckDuckGo instant answer API.
    Returns a dict with company facts for cover letter personalization.
    """
    if not company_name:
        return {"success": False, "facts": [], "error": "No company name found in job description."}

    try:
        # DuckDuckGo instant answer — no API key needed
        url = "https://api.duckduckgo.com/"
        params = {
            "q": company_name,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        response = requests.get(url, params=params, headers=HEADERS, timeout=8)
        response.raise_for_status()
        data = response.json()

        facts = []

        # Abstract (Wikipedia summary)
        abstract = data.get("AbstractText", "").strip()
        if abstract and len(abstract) > 50:
            # Take first two sentences
            sentences = re.split(r'(?<=[.!?])\s+', abstract)
            facts.append(" ".join(sentences[:2]))

        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            text = topic.get("Text", "").strip()
            if text and len(text) > 30:
                facts.append(text)
                if len(facts) >= 3:
                    break

        if facts:
            return {
                "success": True,
                "company_name": company_name,
                "facts": facts[:3],
                "error": ""
            }
        else:
            return {
                "success": False,
                "company_name": company_name,
                "facts": [],
                "error": f"No public information found for '{company_name}'. You may want to research them manually."
            }

    except Exception as e:
        return {
            "success": False,
            "company_name": company_name,
            "facts": [],
            "error": f"Research failed: {str(e)}"
        }