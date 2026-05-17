import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model='basic', llm_model=None):
        self.model = model
        self.llm_model = llm_model
        # load sentence-transformers model once at init
        if model == 'basic':
            self._st_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embeddings(self, text, model):
        if self.model == "basic":
            return self.embed_basic(text)
        elif self.model == "ollama":
            return self.embed_ollama(text)
        elif self.model == "openai":
            return self.embed_openai(text)
        else:
            raise ValueError(f"Unknown model type: {self.model}")

    def embed_basic(self, text):
        if not text or not text.strip():
            return np.zeros(384)  # all-MiniLM-L6-v2 outputs 384 dimensions
        embedding = self._st_model.encode(text)
        return embedding

    def embed_ollama(self, text):
        pass

    def embed_openai(self, text):
        pass