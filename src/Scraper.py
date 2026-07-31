# src/Scraper.py
"""Fetch a job description from a posting URL.

Reality check on what's achievable with plain HTTP:

  * Most sites work — company career pages, Greenhouse, Lever, Workday,
    Ashby, SmartRecruiters, Bayt and similar.
  * LinkedIn works via its own public guest endpoint, which serves posting
    content without a login. We rewrite /jobs/view/<id> to that endpoint.
  * Indeed does not. It sits behind a Cloudflare bot challenge that returns
    403 with `cf-mitigated: challenge` regardless of headers. Only a real
    browser clears it — see the optional Playwright fallback below.

Fetching is one page at a time, on a URL the user explicitly pasted and could
open themselves. There is no crawling and no evasion beyond identifying as a
normal browser.
"""

import json
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# Tags unlikely to contain job description content
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]

# Containers used by the common applicant-tracking systems, most specific first.
ATS_SELECTORS = [
    "#jobDescriptionText",                    # Indeed
    ".show-more-less-html__markup",           # LinkedIn guest view
    ".description__text",                     # LinkedIn guest view (wrapper)
    "#content .job__description",             # Greenhouse
    "#content",                               # Greenhouse (older boards)
    ".posting-page .section-wrapper",         # Lever
    "div[data-automation-id='jobPostingDescription']",  # Workday
    ".job-details-content",                   # SmartRecruiters
    "._content_ud4nd_71",                     # Ashby
    ".jobDescriptionContent",                 # Glassdoor
    "[data-testid='jobDescriptionText']",
]

# Phrases that mean we were served an interstitial, not the posting.
CHALLENGE_MARKERS = (
    "just a moment",
    "verify you are human",
    "checking your browser",
    "enable javascript and cookies",
    "attention required",
    "access to this page has been denied",
)

MIN_USEFUL_CHARS = 200


# ── Site-specific handling ───────────────────────────────────────────────────

def _linkedin_job_id(url):
    """Pull the numeric job id out of any of LinkedIn's URL shapes."""
    for pattern in (
        r"currentJobId=(\d{6,})",
        r"/jobs/view/(?:[^/?#]*?-)?(\d{6,})",
        r"/jobPosting/(\d{6,})",
    ):
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _fetch_linkedin(job_id, timeout):
    """LinkedIn's guest endpoint serves posting content without a login.

    Hitting /jobs/view/<id> directly as a logged-out client redirects to a
    sign-in wall or an unrelated search page; this endpoint does not.
    """
    url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in NOISE_TAGS:
        for el in soup.find_all(tag):
            el.decompose()

    parts = []
    title = soup.select_one(".top-card-layout__title, .topcard__title")
    company_el = soup.select_one(".topcard__org-name-link, .topcard__flavor")
    company = company_el.get_text(strip=True) if company_el else ""
    if title:
        parts.append(title.get_text(strip=True))
    if company:
        parts.append(company)

    body = soup.select_one(".show-more-less-html__markup, .description__text")
    if body:
        parts.append(body.get_text(separator="\n", strip=True))
    else:
        parts.append(soup.get_text(separator="\n", strip=True))

    return _clean_text("\n\n".join(p for p in parts if p)), company


# ── Optional browser fallback ────────────────────────────────────────────────

def browser_available():
    """True only if Playwright AND a downloaded browser binary are present."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


def _fetch_with_browser(url, timeout):
    """Render the page in headless Chromium.

    This is what gets past a Cloudflare JS challenge — the challenge exists
    precisely to tell a script apart from a browser, and here we genuinely are
    one. Returns None if Playwright or its browser binary isn't installed, so
    the caller can fall through to an honest error message.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=HEADERS["User-Agent"],
                locale="en-US",
                viewport={"width": 1366, "height": 900},
            )
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            # Cloudflare's interstitial resolves itself after a few seconds.
            page.wait_for_timeout(5000)
            html = page.content()
            browser.close()
    except Exception:
        return None

    text = _extract_from_html(html)
    return text if text and len(text) >= MIN_USEFUL_CHARS else None


# ── Extraction ───────────────────────────────────────────────────────────────

def _from_json_ld(soup):
    """Read a schema.org JobPosting block, if the site publishes one.

    Workday, SmartRecruiters and many corporate career pages do; Greenhouse and
    Lever do not. Worth trying first when present — it's the description the
    site itself declares, with no surrounding page furniture.
    """
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        try:
            data = json.loads(raw.strip())
        except (json.JSONDecodeError, AttributeError):
            continue

        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("@graph"), list):
                candidates.extend(item["@graph"])
            if item.get("@type") != "JobPosting":
                continue

            parts = [item.get("title") or ""]
            org = item.get("hiringOrganization")
            if isinstance(org, dict):
                parts.append(org.get("name") or "")
            description = item.get("description") or ""
            # description is HTML-in-JSON on nearly every site that uses this
            parts.append(BeautifulSoup(description, "html.parser").get_text(separator="\n"))
            text = _clean_text("\n\n".join(p for p in parts if p))
            if len(text) >= MIN_USEFUL_CHARS:
                return text
    return None


def _extract_from_html(html):
    """Run the extraction strategies in order of decreasing confidence."""
    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: the site's own structured data
    structured = _from_json_ld(soup)
    if structured:
        return structured

    for tag in NOISE_TAGS:
        for el in soup.find_all(tag):
            el.decompose()

    # Strategy 2: a known ATS container
    for selector in ATS_SELECTORS:
        try:
            el = soup.select_one(selector)
        except Exception:
            continue
        if el:
            text = _clean_text(el.get_text(separator="\n", strip=True))
            if len(text) >= MIN_USEFUL_CHARS:
                return text

    # Strategy 3: an id or class that looks description-shaped
    keywords = ["job-description", "jobdescription", "job_description",
                "description", "job-detail", "job-content", "posting"]

    def id_match(value):
        return value and any(k in value.lower() for k in keywords)

    def class_match(value):
        return value and any(k in " ".join(value).lower() for k in keywords)

    for finder in (soup.find(id=id_match), soup.find(class_=class_match)):
        if finder:
            text = _clean_text(finder.get_text(separator="\n", strip=True))
            if len(text) >= MIN_USEFUL_CHARS:
                return text

    # Strategy 4: semantic landmarks, then the whole body
    for tag in ("main", "article", "body"):
        el = soup.find(tag)
        if el:
            text = _clean_text(el.get_text(separator="\n", strip=True))
            if len(text) >= MIN_USEFUL_CHARS:
                return text

    return ""


def company_from_html(html):
    """Read the employer name the page itself declares.

    Far more reliable than regexing prose for "About <Something>" — that
    approach turned a JPMorganChase posting into 'US\\nJ.P'. Sites state the
    employer in structured data or Open Graph tags; use that when they do.
    """
    soup = BeautifulSoup(html, "html.parser")

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads((script.string or script.get_text() or "").strip())
        except (json.JSONDecodeError, AttributeError):
            continue
        for item in (data if isinstance(data, list) else [data]):
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                org = item.get("hiringOrganization")
                if isinstance(org, dict) and org.get("name"):
                    return str(org["name"]).strip()
                if isinstance(org, str) and org.strip():
                    return org.strip()

    for attrs in ({"property": "og:site_name"}, {"name": "author"},
                  {"property": "og:title"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            value = tag["content"].strip()
            # og:title is often "Job Title at Company" or "Job Title - Company"
            m = re.search(r"\s+(?:at|@|\||-|–)\s+([^|\-–]{2,60})$", value)
            if attrs.get("property") == "og:title":
                if m:
                    return m.group(1).strip()
                continue
            if value:
                return value
    return ""


def _clean_text(text):
    """Collapse whitespace and drop repeated navigation cruft.

    The previous version split on "\\n" after extracting with a space
    separator — so there were no newlines to split on and the line filter did
    nothing. Extraction now uses a newline separator, which makes this work.
    """
    lines = []
    seen = set()
    for raw in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            continue
        # Nav/menu leftovers repeat verbatim; keep the first occurrence only.
        key = line.lower()
        if len(line) < 40 and key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines).strip()


# ── Public entry point ───────────────────────────────────────────────────────

def _fail(error, hint=""):
    return {"success": False, "text": "", "error": error, "hint": hint, "company": ""}


def _ok(text, company=""):
    return {"success": True, "text": text, "error": "", "hint": "", "company": company}


def scrape_job_url(url: str, timeout: int = 15) -> dict:
    """
    Scrape a job posting URL and return extracted text.
    Returns:
        {
            "success": bool,
            "text": str,       # extracted job description text
            "error": str,      # message if success=False
            "hint": str,       # what the user can do about it
        }
    """
    url = (url or "").strip()
    if not url:
        return _fail("No URL provided.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    host = re.sub(r"^www\.", "", urlparse(url).netloc.lower())

    # LinkedIn: go straight to the endpoint that serves logged-out visitors.
    if "linkedin.com" in host:
        job_id = _linkedin_job_id(url)
        if not job_id:
            return _fail(
                "That LinkedIn link doesn't contain a job ID.",
                "Open the posting itself so the address looks like "
                "linkedin.com/jobs/view/1234567890, then copy that.",
            )
        try:
            text, company = _fetch_linkedin(job_id, timeout)
            if len(text) >= MIN_USEFUL_CHARS:
                return _ok(text, company)
            return _fail(
                "LinkedIn returned the posting but it had almost no text.",
                "The listing may have expired. Paste the description manually.",
            )
        except Exception as e:
            return _fail(
                f"Could not load that LinkedIn posting: {e}",
                "Paste the description manually below.",
            )

    # Everything else: normal fetch.
    html = None
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            html = response.text
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return _fail("The site took too long to respond.", "Paste the job description manually below.")
    except requests.exceptions.ConnectionError:
        return _fail("Could not connect to that URL.", "Check the link, or paste the description manually.")
    except requests.exceptions.HTTPError as e:
        # `if e.response` is False for ANY error status, because requests makes
        # a Response falsy when not ok — which is why every failure used to
        # report "HTTP error unknown". Compare against None instead.
        code = e.response.status_code if e.response is not None else None
        blocked = e.response is not None and (
            e.response.headers.get("cf-mitigated") == "challenge"
            or e.response.headers.get("server", "").lower() == "cloudflare"
        )

        if code in (403, 429, 503) or blocked:
            text = _fetch_with_browser(url, timeout)
            if text:
                return _ok(text)
            site = host or "this site"
            if "indeed." in host:
                return _fail(
                    "Indeed blocks automated access with a Cloudflare bot check.",
                    "Open the posting in your browser, select the description, "
                    "copy it, and use 'Paste text' above. "
                    + ("" if browser_available() else
                       "(A headless browser could fetch it automatically — "
                       "run `playwright install chromium` to enable that.)"),
                )
            return _fail(
                f"{site} refused automated access (HTTP {code}).",
                "Paste the job description manually below.",
            )
        if code == 404:
            return _fail("That posting no longer exists (HTTP 404).", "The listing may have been taken down.")
        return _fail(
            f"The site returned HTTP {code}." if code else f"Request failed: {e}",
            "Paste the job description manually below.",
        )
    except Exception as e:
        return _fail(f"Unexpected error: {e}", "Paste the job description manually below.")

    html = html or ""
    text = _extract_from_html(html)

    # A 200 that's really an interstitial: retry in a real browser if we can.
    if len(text) < MIN_USEFUL_CHARS or any(m in text[:400].lower() for m in CHALLENGE_MARKERS):
        rendered = _fetch_with_browser(url, timeout)
        if rendered:
            return _ok(rendered)

    if len(text) < MIN_USEFUL_CHARS:
        return _fail(
            "The page loaded, but no job description could be found in it.",
            "The description is probably loaded by JavaScript. Paste it manually below.",
        )

    return _ok(text, company_from_html(html))
