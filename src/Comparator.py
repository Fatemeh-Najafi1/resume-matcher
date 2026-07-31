import numpy as np

from .utils import score_band, score_verdict


class Comparator:
    """Scores how close two documents sit in embedding space."""

    def __init__(self, resume, job, model="basic", llmclient=None):
        self.resume = resume
        self.job = job
        self.model = model
        self.llmclient = llmclient

    def calculate_similarity(self, vec1, vec2):
        if self.model == "basic":
            return self._compare_basic(vec1, vec2)
        elif self.model == "openai":
            return self._compare_openai(vec1, vec2)
        elif self.model == "ollama":
            return self._compare_ollama(vec1, vec2)
        else:
            raise ValueError(f"Unknown comparison model: {self.model}")

    def _compare_basic(self, vec1, vec2):
        """Cosine similarity, computed explicitly.

        The vectors come back from .encode() unnormalized, so we divide by both
        norms rather than assuming unit length. Measuring the angle — not the
        distance — means a short, precise resume isn't penalised against a long
        job posting just for having fewer words.
        """
        v1, v2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            # one of the documents was empty — no meaningful angle exists
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _compare_openai(self, vec1, vec2):
        raise NotImplementedError("OpenAI comparison is not wired up yet.")

    def _compare_ollama(self, vec1, vec2):
        raise NotImplementedError("Ollama comparison is not wired up yet.")

    def provide_insight(self, vec1, vec2, score, llmclient=None):
        client = llmclient or self.llmclient
        if client is None:
            return self.basic_insight(vec1, vec2, score)
        raise NotImplementedError(f"LLM insight via '{client}' is not wired up yet.")

    def basic_insight(self, vec1, vec2, score):
        """Plain-English reading of the score, using the shared bands."""
        return {
            "score": float(score),
            "band": score_band(score),
            "interpretation": score_verdict(score),
        }
