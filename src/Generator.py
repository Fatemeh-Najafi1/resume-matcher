import re 
class Generator:
    def __init__(self,  model = 'basic' , llmclient = None ): 
        self.model = model 
        self.llmclient = llmclient

    def generate_text(self, resume, job, score ):
        if self.model == "basic":
            return self._generate_basic(resume, job, score)
        elif self.model == "openai":
            return self._generate_openai(resume, job, score)
        elif self.model == "ollama":
            return self._generate_ollama(resume, job, score)
        else:
            raise ValueError(f"Unknown generator model: {self.model}")

    def _generate_basic(self, resume, job, score):
        #finding keywordds in job and reume
        job_keywords = set(re.findall(r"\b[a-zA-Z]{3,}\b", job.lower()))
        resume_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", resume.lower()))
        #words in job but not resume
        missing = job_keywords - resume_words
         # intelligently insert missing terms into summary or skills
        rewritten_resume = resume
        if "SKILLS" in resume.upper():
         rewritten_resume = re.sub(
            r"(SKILLS[\s\S]*?)(\n[A-Z ]{3,}|$)",
            lambda m: m.group(1)
            + f"\n• Additional relevant skills: {missing}\n"
            + (m.group(2) if m.group(2) else ""),
            rewritten_resume,
            flags=re.IGNORECASE,
        )
        else:
        # if no SKILLS section found, add to end
          rewritten_resume += f"\n\nAdditional relevant skills: {missing}\n"
        suggestion = "\n\n---\n🔧 Suggested Resume Update:\n"
        suggestion += f"Match score: {round(score, 2)}\n"
        suggestion += "You may consider including these missing terms: "
        suggestion += ", ".join(list(missing)[:20])  # limit output

        return resume + suggestion

        
    def _generate_ollama(resume, job, score):
        #later phase development
        pass 
    def _generate_openai(resume, job, score):
        #later phase develoment 
        pass 


