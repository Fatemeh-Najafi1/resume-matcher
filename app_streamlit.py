# app_streamlit.py
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from src.Reader import Reader, UnreadableDocumentError
from src.Cleaner import Cleaner
from src.Embedder import Embedder
from src.Comparator import Comparator
from src.Generator import Generator
from src.Scraper import scrape_job_url
from src.LLMAnalyzer import analyze
from src.CompanyResearcher import extract_company_name, research_company
from src.Requirements import extract_requirements
from src.Matcher import split_bullets, match, unmatched_bullets
from src import Verdict, Fidelity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide"
)


# ── Cached network calls ──────────────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction. Without these caches
# a single keystroke would re-scrape the job posting and re-hit DuckDuckGo.
@st.cache_data(show_spinner=False, ttl=3600)
def cached_scrape(url):
    return scrape_job_url(url)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_research(company_name):
    return research_company(company_name)


def get_key(name):
    """Read an API key from .env locally, or Streamlit secrets when deployed.

    .env is gitignored, so it does not exist on Streamlit Community Cloud — a
    deployment that only reads os.getenv finds nothing and every visitor sees
    "no API key". Secrets set in the Streamlit dashboard are read here instead.
    """
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return st.secrets.get(name, "")
    except Exception:
        # No secrets.toml configured — normal when running locally.
        return ""


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 Resume Matcher")
st.markdown(
    "Upload your resume and a job description. Get a similarity score, "
    "an honest skills gap report, and AI-powered resume suggestions — "
    "**without adding skills you don't have.**"
)
st.divider()

# ── Sidebar: LLM settings ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ AI Settings")
    st.markdown("Choose your AI provider for the analysis report.")

    provider = st.selectbox(
        "AI Provider",
        options=["gemini", "openai", "anthropic"],
        format_func=lambda x: {
            "gemini": "🟢 Google Gemini",
            "openai": "⚫ OpenAI GPT-4o-mini",
            "anthropic": "🟠 Anthropic Claude"
        }[x]
    )

    # Pre-fill from .env if a key is present, so the app works out of the box.
    default_keys = {
        "gemini": get_key("GEMINI_API_KEY"),
        "openai": get_key("OPENAI_API_KEY"),
        "anthropic": get_key("ANTHROPIC_API_KEY"),
    }
    default_key = default_keys[provider]

    api_key = st.text_input(
        "API Key",
        type="password",
        value=default_key,
        placeholder={
            "gemini": "AIza...",
            "openai": "sk-...",
            "anthropic": "sk-ant-..."
        }[provider],
        help="Read from your .env file if set. You can override it here."
    )

    key_source = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider]
    if default_key:
        st.caption(f"✓ Loaded from `{key_source}`")
    else:
        st.caption(f"No `{key_source}` configured — paste a key above to enable AI analysis.")

    if provider == "gemini":
        st.info("Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com)", icon="ℹ️")
    elif provider == "openai":
        st.info("Get an OpenAI key at [platform.openai.com](https://platform.openai.com)", icon="ℹ️")
    elif provider == "anthropic":
        st.info("Get an Anthropic key at [console.anthropic.com](https://console.anthropic.com)", icon="ℹ️")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown(
        "1. Upload your resume\n"
        "2. Provide the job description\n"
        "3. Get your match score\n"
        "4. Get AI analysis\n"
        "5. Research the company"
    )

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

# ── Resume input ──────────────────────────────────────────────────────────────
with col1:
    st.subheader("📋 Your Resume")
    resume_file = st.file_uploader(
        "Upload resume (TXT, DOCX, or PDF)",
        type=["txt", "docx", "pdf"],
        key="resume_upload"
    )
    st.caption("PDFs must contain real text — a scan or photo has nothing to extract.")
    resume_text_input = st.text_area(
        "Or paste your resume here",
        height=200,
        placeholder="Paste resume text...",
        key="resume_text"
    )

# ── Job description input ─────────────────────────────────────────────────────
with col2:
    st.subheader("💼 Job Description")

    job_input_method = st.radio(
        "How would you like to provide the job description?",
        options=["🔗 Paste a URL", "📝 Paste text", "📁 Upload a file"],
        horizontal=True
    )

    job_text = ""
    scraped_company = ""   # the employer the page declares, if we scraped one

    if job_input_method == "🔗 Paste a URL":
        job_url = st.text_input(
            "Job posting URL",
            placeholder="https://careers.example.com/job/12345"
        )
        if job_url:
            with st.spinner("Scraping job description..."):
                result = cached_scrape(job_url)
            if result["success"]:
                job_text = result["text"]
                scraped_company = result.get("company", "")
                msg = f"✅ Extracted {len(job_text.split())} words from the page."
                if scraped_company:
                    msg += f" Employer: {scraped_company}."
                st.success(msg)
                with st.expander("Preview extracted text"):
                    st.text(job_text[:1000] + ("..." if len(job_text) > 1000 else ""))
            else:
                st.warning(f"⚠️ {result['error']}")
                if result.get("hint"):
                    st.info(result["hint"])
                job_text = st.text_area(
                    "Paste the job description here instead",
                    height=200,
                    placeholder="Select the description on the posting page, copy, and paste here...",
                    key="job_fallback",
                )

    elif job_input_method == "📝 Paste text":
        job_text = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="Paste the full job description...",
            key="job_paste"
        )

    elif job_input_method == "📁 Upload a file":
        job_file = st.file_uploader(
            "Upload job description (TXT, DOCX, or PDF)",
            type=["txt", "docx", "pdf"],
            key="job_upload"
        )
        if job_file:
            try:
                job_text = Reader().load_job_description(job_file)
                st.success(f"✅ Loaded {len(job_text.split())} words from file.")
            except (UnreadableDocumentError, ValueError) as e:
                st.error(f"Couldn't read that file: {e}")

st.divider()

# ── Run analysis ──────────────────────────────────────────────────────────────
run_btn = st.button("🚀 Analyse My Resume", type="primary", use_container_width=True)

if run_btn:
    # Resolve resume text
    resume_text = ""
    if resume_file:
        try:
            resume_text = Reader().load_resume(resume_file)
        except (UnreadableDocumentError, ValueError) as e:
            st.error(f"Couldn't read your resume: {e}")
            st.stop()
    elif resume_text_input.strip():
        resume_text = resume_text_input.strip()

    # Validate inputs
    if not resume_text:
        st.error("Please upload or paste your resume.")
        st.stop()
    if not job_text:
        st.error("Please provide a job description.")
        st.stop()

    # Step 1: similarity score
    with st.spinner("Computing similarity score..."):
        c = Cleaner()
        resume_clean = c.preprocess(resume_text, doc_type="resume")
        job_clean = c.preprocess(job_text, doc_type="job")

        # Picks the multilingual encoder if either document contains Arabic;
        # the English-only model would return noise rather than fail.
        e = Embedder.for_documents(resume_text, job_text)
        resume_vect = e.get_embeddings(resume_clean["clean_text"])
        job_vect = e.get_embeddings(job_clean["clean_text"])

        comp = Comparator(resume_text, job_text, model="basic")
        score = comp.calculate_similarity(resume_vect, job_vect)
        insight = comp.provide_insight(resume_vect, job_vect, score)

    # Step 2: keyword gap (no LLM involved)
    basic_report = Generator().generate_text(resume_clean, job_clean, score)

    # Step 2b: requirement-level coverage — the analysis that can be acted on
    with st.spinner("Extracting requirements and matching them to your resume..."):
        requirements, req_source = extract_requirements(
            job_text, provider=provider if api_key else None, api_key=api_key
        )
        bullets = split_bullets(resume_text)
        coverage_result = match(requirements, bullets, e, resume_text=resume_text)

    # Step 3: LLM analysis
    llm_result = None
    if api_key:
        with st.spinner(f"Analysing with {provider}... this may take 15-30 seconds."):
            llm_result = analyze(resume_text, job_text, provider=provider, api_key=api_key)

    # Step 4: company research. Prefer the name the page declared over a
    # regex guess at the prose.
    company_name = scraped_company or extract_company_name(job_text)
    research = None
    if company_name:
        with st.spinner(f"Researching {company_name}..."):
            research = cached_research(company_name)

    # Persist so the results survive the re-run that any widget click triggers.
    st.session_state["results"] = {
        "resume_text": resume_text,
        "job_text": job_text,
        "resume_clean": resume_clean,
        "job_clean": job_clean,
        "score": score,
        "insight": insight,
        "basic_report": basic_report,
        "coverage_result": coverage_result,
        "req_source": req_source,
        "llm_result": llm_result,
        "company_name": company_name,
        "research": research,
        "had_key": bool(api_key),
    }

# ── Results ───────────────────────────────────────────────────────────────────
results = st.session_state.get("results")

if results:
    score = results["score"]
    insight = results["insight"]

    st.subheader("📊 Semantic Match Score")
    st.caption(
        "Cosine similarity between your resume and the posting, measured in "
        "embedding space. It reflects how closely the two documents talk about "
        "the same things — not how qualified you are."
    )

    score_pct = round(float(score) * 100, 1)
    col_score, col_insight = st.columns([1, 3])
    with col_score:
        color = "green" if score_pct >= 70 else ("orange" if score_pct >= 55 else "red")
        st.markdown(
            f"<h1 style='color:{color}; text-align:center'>{score_pct}%</h1>",
            unsafe_allow_html=True
        )
    with col_insight:
        st.info(insight.get("interpretation", ""))

        shared = sorted(set(results["resume_clean"]["skills"]) & set(results["job_clean"]["skills"]))
        gap = sorted(set(results["job_clean"]["skills"]) - set(results["resume_clean"]["skills"]))
        if shared:
            st.markdown("**Skills you share with this posting:** " + ", ".join(shared))
        if gap:
            st.markdown("**Skills the posting wants that your resume doesn't show:** " + ", ".join(gap))

    with st.expander("📝 Full keyword gap report (no AI — literal word comparison)"):
        st.text(results["basic_report"])

    st.divider()

    # ── Requirement coverage ──────────────────────────────────────────────────
    cov = results.get("coverage_result")
    if cov and cov["requirements"]:
        counts = cov["counts"]
        total = len(cov["requirements"])

        st.subheader("🎯 Requirement Coverage")
        st.caption(
            "Each requirement in the posting, matched against the individual lines "
            "of your resume. Unlike the score above, this tells you *which* "
            "requirement went unanswered and *which* line is your best evidence."
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Coverage", f"{cov['coverage']*100:.0f}%",
                  help="Covered requirements, partials counted as half. Personal-quality "
                       "requirements are excluded from the denominator entirely — "
                       "counting them either way would be a guess dressed up as a "
                       "measurement.")
        m2.metric("✅ Covered", counts.get(Verdict.COVERED, 0))
        m3.metric("🟡 Partial", counts.get(Verdict.PARTIAL, 0))
        m4.metric("❓ Not shown", counts.get(Verdict.UNSUPPORTED, 0),
                  help="Your resume doesn't speak to these. That is not the same as not "
                       "having them — add them if you do.")
        m5.metric("🚫 Ruled out", counts.get(Verdict.CONTRADICTED, 0),
                  help="Your resume states something that cannot meet the demand — a "
                       "degree below the floor, or fewer years than asked. These are the "
                       "only ones that are genuine mismatches rather than gaps.")

        if counts.get(Verdict.JUDGEMENT, 0):
            st.caption(
                f"🤔 {counts[Verdict.JUDGEMENT]} further requirement(s) ask about personal "
                "qualities a resume cannot settle. They get no verdict and are excluded "
                "from the coverage figure — scroll down to judge them yourself."
            )

        if counts.get(Verdict.UNSUPPORTED, 0) and not counts.get(Verdict.CONTRADICTED, 0):
            st.success(
                "Nothing here rules you out. Every gap below is something your resume "
                "simply doesn't mention — which is a writing problem, not a "
                "qualification problem.",
                icon="💡",
            )

        if results.get("req_source") == "rules":
            st.info(
                "Requirements were extracted without AI (no API key), so the list is "
                "rougher. Add a key in the sidebar for cleaner extraction.",
                icon="ℹ️",
            )

        badge = {Verdict.COVERED: "✅", Verdict.PARTIAL: "🟡",
                 Verdict.UNSUPPORTED: "❓", Verdict.CONTRADICTED: "🚫",
                 Verdict.JUDGEMENT: "🤔"}
        for row in cov["requirements"]:
            kind = "" if row.get("kind") == "required" else " · preferred"
            header = f"{badge.get(row['status'], '·')} {row['text']}{kind}"
            with st.expander(header, expanded=False):
                if row.get("reason"):
                    st.markdown(f"**Verdict:** {row['reason']}")
                if row.get("decided_by") == "judgement":
                    st.caption(
                        "No verdict on purpose. This asks about a personal quality, which a "
                        "resume can't settle — and on this class of requirement two careful "
                        "readers disagree, so any automatic answer would be a coin flip "
                        "dressed up as a measurement. Your closest evidence is below."
                    )
                elif row.get("decided_by") == "literal":
                    st.caption(
                        "Decided by checking the words themselves, not by similarity — "
                        "naming a specific technology or degree is something we can verify."
                    )
                else:
                    engine = cov.get("scorer", "bi-encoder")
                    st.caption(
                        f"Decided by semantic similarity ({engine}, score {row['score']:.2f}). "
                        "This one is too general to check literally, so treat it as a "
                        "suggestion and judge it yourself — measured precision on this "
                        "class of decision is about 0.9, not 1.0."
                    )
                st.caption(Verdict.MEANING.get(row["status"], ""))

                if row["evidence"]:
                    st.markdown("**Your best evidence:**")
                    st.info(row["evidence"])
                elif row["status"] == Verdict.UNSUPPORTED:
                    st.markdown(
                        "_Your resume does not mention this. That is not the same as not "
                        "having it — if you do, say so explicitly._"
                    )
                elif row["status"] == Verdict.CONTRADICTED:
                    st.markdown(
                        "_This is a real shortfall rather than a gap in the writing._"
                    )

        idle = unmatched_bullets(cov)
        if idle:
            with st.expander(f"💤 {len(idle)} resume lines that matched nothing in this posting"):
                st.caption(
                    "Not necessarily weak lines — just the ones earning least against "
                    "*this* role. Worth trimming if you need the space."
                )
                for b in idle:
                    st.markdown(f"• {b}")

        st.divider()

    # ── LLM analysis ──────────────────────────────────────────────────────────
    st.subheader("🤖 AI-Powered Analysis")

    llm_result = results["llm_result"]

    if not results["had_key"]:
        st.warning("No API key found. Add your key in the sidebar to get the full AI analysis.")
    elif llm_result is None or not llm_result.get("success"):
        err = (llm_result or {}).get("error", "Unknown error")
        st.error(f"AI analysis failed: {err}")
    else:
        # The LLM produces its OWN match score from reading both documents. It is
        # a different measurement from the cosine score above — that one compares
        # vectors, this one is a judgement. Showing them side by side, each with
        # its own label, avoids the earlier bug where the LLM's explanation sat
        # under the embedding score and appeared to be explaining it.
        llm_score = llm_result.get("match_score")
        if llm_score is not None:
            col_a, col_b = st.columns(2)
            col_a.metric("Semantic similarity", f"{score_pct}%", help="Cosine similarity of embeddings")
            col_b.metric(f"{provider.title()} match score", f"{llm_score}%", help="The model's own judgement")
            st.caption(
                "These two can disagree — the first measures vocabulary and topic overlap, "
                "the second is a judgement about your fit. A large gap is itself informative."
            )

        st.markdown(f"#### Why {provider.title()} gave that score")
        st.write(llm_result.get("score_explanation", ""))

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### ✅ Highlight These")
            st.caption("Skills you have — mention them more prominently")
            items = llm_result.get("skills_to_highlight", [])
            for item in items:
                st.markdown(f"• {item}")
            if not items:
                st.caption("None identified")

        with c2:
            st.markdown("#### ➕ Add to Resume")
            st.caption("Skills you likely have but didn't mention")
            items = llm_result.get("skills_to_add", [])
            for item in items:
                st.markdown(f"• {item}")
            if not items:
                st.caption("None identified")

        with c3:
            st.markdown("#### 📚 Develop These")
            st.caption("Genuine gaps — don't add these until you have them")
            items = llm_result.get("skills_to_develop", [])
            for item in items:
                st.markdown(f"• {item}")
            if not items:
                st.caption("None identified")

        bullets = llm_result.get("rewritten_bullets", [])
        if bullets:
            st.divider()
            st.markdown("#### ✏️ Suggested Bullet Rewrites")

            # The model was told not to invent anything. That instruction is a
            # sentence in a prompt, so every rewrite is checked against the
            # resume afterwards rather than trusted. See src/Fidelity.py.
            screened = Fidelity.screen_rewrites(bullets, results.get("resume_text", ""))
            summary = Fidelity.summarise(screened)
            if screened and any(not s_["fidelity"]["ok"] for s_ in screened):
                st.error(summary, icon="🚨")
            elif summary:
                st.success(summary, icon="✅")
            st.caption(
                "Each rewrite is checked automatically for skills, figures and "
                "seniority claims your resume never made. Use one only if it is "
                "true of you — the check catches inventions, not exaggerations."
            )

            for b in screened:
                fid = b["fidelity"]
                mark = "✅" if fid["ok"] else "🚨"
                with st.expander(f"{mark} Original: {b.get('original', '')[:70]}..."):
                    st.markdown(f"**Rewritten:** {b.get('rewritten', '')}")
                    st.caption(f"Why: {b.get('reason', '')}")
                    if not fid["ok"]:
                        st.error(
                            "**This rewrite adds something your resume does not say.** "
                            "Claiming it would not survive an interview:",
                            icon="🚨",
                        )
                    for problem in fid["problems"]:
                        st.markdown(f"- {problem}")

        advice = llm_result.get("overall_advice", "")
        if advice:
            st.divider()
            st.markdown("#### 💡 Overall Advice")
            st.info(advice)

    st.divider()

    # ── Company research ──────────────────────────────────────────────────────
    st.subheader("🔍 Company Research")
    st.caption("Use these facts to personalize your cover letter")

    company_name = results["company_name"]
    research = results["research"]

    if not company_name:
        st.info(
            "Could not automatically detect the company name. "
            "Research them manually for cover letter personalization."
        )
    elif research and research.get("success") and research.get("facts"):
        st.markdown(f"**{company_name}**")
        for fact in research["facts"]:
            st.markdown(f"• {fact}")
        st.caption(
            "Tip: reference one of these facts in your cover letter opening "
            "to show you've done your homework."
        )
    else:
        st.warning(
            f"Could not find public information for '{company_name}'. "
            "Try searching them manually on LinkedIn or their website."
        )

    st.divider()

    col_r, col_j = st.columns(2)
    with col_r:
        with st.expander("📄 Your Resume"):
            st.text_area(
                "Resume text", results["resume_text"], height=300,
                key="resume_preview", label_visibility="collapsed"
            )
    with col_j:
        with st.expander("💼 Job Description"):
            st.text_area(
                "Job description text", results["job_text"], height=300,
                key="job_preview", label_visibility="collapsed"
            )
