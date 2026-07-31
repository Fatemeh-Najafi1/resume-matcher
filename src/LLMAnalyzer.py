# src/LLMAnalyzer.py
import json
import re

from .config import LLM_MODELS

# ── Supported providers ──────────────────────────────────────────────────────
# "gemini"    → Google Gemini (free tier, default)
# "openai"    → OpenAI GPT-4o
# "anthropic" → Anthropic Claude

SYSTEM_PROMPT = """You are a professional career coach and resume expert.
Your job is to analyze a resume against a job description and produce a structured, honest report.

CRITICAL RULES:
- NEVER suggest adding skills the candidate does not already demonstrate in their resume.
- Clearly separate skills they HAVE but haven't mentioned from skills they genuinely need to DEVELOP.
- Be specific and actionable. Vague advice is useless.
- Rewritten bullets must sound human and professional, not generic AI output.
- Return ONLY valid JSON, no markdown, no explanation outside the JSON."""

ANALYSIS_PROMPT = """Analyze this resume against the job description and return a JSON object with exactly these fields:

{{
  "match_score": <integer 0-100>,
  "score_explanation": "<2-3 sentences explaining WHY this score, citing specific strengths and gaps>",
  "skills_to_highlight": ["<skill already in resume that matches job but could be emphasized more>", ...],
  "skills_to_add": ["<skill candidate likely has based on their experience but didn't mention>", ...],
  "skills_to_develop": ["<skill the job requires that candidate genuinely doesn't seem to have>", ...],
  "rewritten_bullets": [
    {{
      "original": "<original bullet from resume>",
      "rewritten": "<improved version using job description language>",
      "reason": "<one sentence explaining the change>"
    }},
    ...
  ],
  "overall_advice": "<3-4 sentences of honest, specific advice for this application>"
}}

Rules:
- skills_to_highlight: max 5 items
- skills_to_add: max 5 items (only if genuinely inferable from their background)
- skills_to_develop: max 5 items (honest gaps — don't sugarcoat)
- rewritten_bullets: pick the 3 most impactful bullets to rewrite
- Return ONLY the JSON object, nothing else

RESUME:
{resume}

JOB DESCRIPTION:
{job}"""


def analyze(resume_text: str, job_text: str, provider: str = "gemini", api_key: str = "") -> dict:
    """
    Run LLM analysis on resume vs job description.
    Returns a dict with the analysis or an error.
    """
    if not api_key:
        return {"success": False, "error": f"No API key provided for {provider}."}

    prompt = ANALYSIS_PROMPT.format(
        resume=resume_text[:8000],  # cap to avoid token limits
        job=job_text[:4000]
    )
    return call_json(provider, api_key, SYSTEM_PROMPT, prompt)


def call_json(provider: str, api_key: str, system_prompt: str, user_prompt: str) -> dict:
    """Run one JSON-returning prompt against a provider.

    Shared so requirement extraction and the analysis report use the same
    provider dispatch, model config and error handling rather than each
    growing their own copy.
    """
    if not api_key:
        return {"success": False, "error": f"No API key provided for {provider}."}
    try:
        if provider == "gemini":
            return _run_gemini(api_key, system_prompt, user_prompt)
        elif provider == "openai":
            return _run_openai(api_key, system_prompt, user_prompt)
        elif provider == "anthropic":
            return _run_anthropic(api_key, system_prompt, user_prompt)
        else:
            return {"success": False, "error": f"Unknown provider: {provider}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _parse_json_response(text: str) -> dict:
    """Extract and parse JSON from LLM response."""
    # Strip markdown code blocks if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            # A prompt that asks for a bare JSON array lands here.
            return {"success": True, "data": data}
        data["success"] = True
        return data
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Could not parse LLM response as JSON: {str(e)}", "raw": text[:500]}


def _run_gemini(api_key: str, system_prompt: str, prompt: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model_name = LLM_MODELS["gemini"]
    try:
        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
        # temperature=0: re-running the same resume against the same posting
        # should give the same requirements, not a differently-split list.
        response = model.generate_content(
            prompt, generation_config={"temperature": 0.0}
        )
    except Exception as e:
        return {"success": False, "error": _gemini_error(e, model_name)}
    return _parse_json_response(response.text)


def _gemini_error(exc, model_name):
    """Turn a Gemini failure into something the user can act on.

    The two failure modes look nothing alike but both used to surface as a raw
    stack-trace string. Note that listing models is NOT proof you may call them
    — gemini-2.5-flash appears in list_models() and still 404s for new keys —
    so we don't offer a list here, just the fix.
    """
    text = str(exc)
    lowered = text.lower()

    if "404" in text or "not found" in lowered or "no longer available" in lowered:
        return (f"The Gemini model '{model_name}' isn't available to this API key "
                f"(Google closes older versions to new users). Set a current model "
                f"under `llm.gemini` in config.yaml — 'gemini-flash-latest' tracks "
                f"whatever the current flash model is.")

    if "429" in text or "quota" in lowered or "resourceexhausted" in lowered:
        return ("Gemini rejected the request for quota reasons — the free tier for "
                "this model may be closed, or you've hit the rate limit. Wait a "
                "minute and retry, or pick a different model under `llm.gemini` "
                "in config.yaml.")

    if "api key" in lowered or "permission" in lowered or "401" in text or "403" in text:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env, or paste a key in the sidebar."

    return f"Gemini request failed: {text[:300]}"


def _run_openai(api_key: str, system_prompt: str, prompt: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODELS["openai"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return _parse_json_response(response.choices[0].message.content)


def _run_anthropic(api_key: str, system_prompt: str, prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=LLM_MODELS["anthropic"],
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    return _parse_json_response(response.content[0].text)