#from config import score_threshold
import numpy as np

class Comperator:
    def __init__(self, resume, job, model='basic', llmclient=None, thershold=0.7):
        self.resume = resume
        self.job = job
        self.model = model
        self.llmclient = llmclient
        self.thershold = thershold  # if theresehole is not None else score_threshold

    def calculate_similarity(self, vec1, vec2):
        if self.model == 'basic':
            if self.model == "basic":
                score = self._compare_basic(vec1, vec2)
            elif self.model == "openai":
                score = self._compare_openai(vec1, vec2)
            elif self.model == "ollama":
                score = self._compare_ollama(vec1, vec2)
            else:
                raise ValueError(f"Unknown comparison model: {self.model}")
            return score

    def _compare_basic(self, vec1, vec2):
        v1, v2 = np.array(vec1), np.array(vec2)
        # get vector 1 and 2 turn them into numpu array for operation
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        # calculating leenghth
        # if they hold no vallues return 0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # computing cosine

    def _compare_openai(vec1, vec2):
        # for develomemt in later phases
        pass

    def _compare_ollama(vec1, vec2):
        # later phases
        pass

    def provide_insight(self, vec1, vec2, score, llmclient=None):
        if llmclient == None:
            insight = self.basic_insight(vec1, vec2, score)
        elif llmclient == 'openai':
            pass
        elif llmclient == 'ollama':
            pass

        return insight

    def basic_insight(self, vec1, vec2, score):
        t = self.thershold
        if score >= t + 0.15:
            comment = "you are ready for applying, wonderful match"
        elif score >= t:
            comment = "good matchyou need to develop aa few more skills to increase you chance"
        elif score >= t - 0.15:
            comment = "Partial match; align you resume with job descrioption."
        else:
            comment = "Low match; revise resume to target the job more closely."

        return {
            "score": score,
            "interpretation": comment
        }

    def llm_openai_insight(self, vec1, vec2, score):
        """(Future) Generate insight via OpenAI GPT model."""
        print("⚠️ OpenAI-based insight not implemented yet.")
        return self.basic_insight(vec1, vec2, score)

    def llm_ollama_insight(self, vec1, vec2, score):
        """(Future) Generate insight via Ollama local model."""
        print("⚠️ Ollama-based insight not implemented yet.")
        return self.basic_insight(vec1, vec2, score)
