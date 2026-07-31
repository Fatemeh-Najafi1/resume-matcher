from .utils import score_verdict


class Generator:
    """Builds the keyword-gap report.

    This step is deliberately dumb: it reports only words that are literally
    absent from the resume. There is no model here, so there is no mechanism
    by which it could invent a skill the candidate doesn't have. That property
    is the whole basis of the "no fabrication" guarantee — it's structural,
    not a promise made to a prompt.
    """

    def __init__(self, model="basic", llmclient=None, max_missing=20):
        self.model = model
        self.llmclient = llmclient
        self.max_missing = max_missing

    def generate_text(self, resume_clean, job_clean, score):
        """resume_clean / job_clean are Cleaner.preprocess() outputs."""
        if self.model == "basic":
            return self._generate_basic(resume_clean, job_clean, score)
        elif self.model == "openai":
            return self._generate_openai(resume_clean, job_clean, score)
        elif self.model == "ollama":
            return self._generate_ollama(resume_clean, job_clean, score)
        else:
            raise ValueError(f"Unknown generator model: {self.model}")

    def _generate_basic(self, resume_clean, job_clean, score):
        # Both sides are lemmatized and stopword-filtered by the Cleaner, so
        # "managed" and "manage" no longer count as a miss — which they did
        # when this step read the raw text and skipped the pipeline.
        job_keywords = set(job_clean["keywords"])
        resume_keywords = set(resume_clean["keywords"])
        counts = job_clean.get("keyword_counts", {})
        # Rank by how often the posting repeats the term, not alphabetically —
        # a requirement mentioned five times outranks a one-off adjective.
        missing = sorted(job_keywords - resume_keywords,
                         key=lambda w: (-counts.get(w, 0), w))

        job_skills = set(job_clean["skills"])
        resume_skills = set(resume_clean["skills"])
        missing_skills = sorted(job_skills - resume_skills)
        matched_skills = sorted(job_skills & resume_skills)

        rule = "=" * 50
        out = [rule, "RESUME MATCH REPORT", rule, ""]
        out.append(f"Match score: {round(float(score) * 100, 1)}%")
        out.append(f"Assessment:  {score_verdict(score)}")
        out.append("")

        if matched_skills:
            out.append(f"Skills you already share with this posting ({len(matched_skills)}):")
            out.extend(f"   + {s}" for s in matched_skills)
            out.append("")

        if missing_skills:
            out.append(f"Skills the posting asks for that your resume doesn't show ({len(missing_skills)}):")
            out.extend(f"   - {s}" for s in missing_skills)
            out.append("")

        if missing:
            shown = missing[: self.max_missing]
            out.append(f"Other job-description terms absent from your resume, "
                       f"most-repeated first (showing {len(shown)} of {len(missing)}):")
            out.extend(
                f"   - {kw}" + (f"  (x{counts[kw]})" if counts.get(kw, 0) > 1 else "")
                for kw in shown
            )
            out.append("")
            out.append("If you genuinely have any of the above, add it using the posting's")
            out.append("exact wording. If you don't, treat it as something to learn —")
            out.append("do not add a skill you haven't got.")
        else:
            out.append("Your resume covers every key term in this job description.")

        out.append("")
        out.append(rule)
        return "\n".join(out)

    def _generate_ollama(self, resume_clean, job_clean, score):
        raise NotImplementedError("Ollama generation is not wired up yet.")

    def _generate_openai(self, resume_clean, job_clean, score):
        raise NotImplementedError("OpenAI generation is not wired up yet.")
