import re

class Generator:
    def __init__(self, model='basic', llmclient=None):
        self.model = model
        self.llmclient = llmclient

    def generate_text(self, resume, job, score):
        if self.model == "basic":
            return self._generate_basic(resume, job, score)
        elif self.model == "openai":
            return self._generate_openai(resume, job, score)
        elif self.model == "ollama":
            return self._generate_ollama(resume, job, score)
        else:
            raise ValueError(f"Unknown generator model: {self.model}")

    def _generate_basic(self, resume, job, score):
        # extract meaningful words (4+ chars) from job and resume
        job_keywords = set(re.findall(r"\b[a-zA-Z]{4,}\b", job.lower()))
        resume_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", resume.lower()))

        # common filler words to ignore
        filler = {
            "that", "this", "with", "from", "have", "will", "your",
            "their", "they", "been", "were", "also", "about", "which",
            "when", "what", "more", "other", "into", "than", "then",
            "each", "such", "both", "through", "those", "these"
        }

        missing = sorted(job_keywords - resume_words - filler)

        # build a clean, readable suggestion block
        suggestion = "\n\n" + "=" * 50
        suggestion += "\n📊 RESUME MATCH REPORT\n"
        suggestion += "=" * 50
        suggestion += f"\n\nMatch Score: {round(float(score) * 100, 1)}%\n"

        if score >= 0.85:
            suggestion += "Assessment: Excellent match — strong candidate for this role.\n"
        elif score >= 0.70:
            suggestion += "Assessment: Good match — a few gaps to address before applying.\n"
        elif score >= 0.55:
            suggestion += "Assessment: Partial match — resume needs better alignment with this role.\n"
        else:
            suggestion += "Assessment: Low match — significant skill gaps exist for this role.\n"

        if missing:
            # split into skills to highlight vs skills to develop
            suggestion += "\n📌 Keywords in job description not found in your resume:\n"
            for kw in missing[:20]:
                suggestion += f"   • {kw}\n"

            suggestion += "\n💡 Tip: If you have experience with any of the above, "
            suggestion += "add them to your resume using the job's exact phrasing.\n"
            suggestion += "If you don't have them yet, consider listing them as skills to develop — "
            suggestion += "do not add skills you haven't learned.\n"
        else:
            suggestion += "\n✅ Your resume covers the key terms in this job description.\n"

        suggestion += "=" * 50

        return resume + suggestion

    def _generate_ollama(self, resume, job, score):
        pass

    def _generate_openai(self, resume, job, score):
        pass