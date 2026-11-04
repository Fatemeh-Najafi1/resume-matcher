from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, EMBEDDING_DIM, MULTILINGUAL_MODEL
from .utils import has_arabic


@lru_cache(maxsize=2)
def load_model(name: str) -> SentenceTransformer:
    """Load a sentence-transformers model once per process.

    Streamlit re-runs the whole script on every interaction, so without this
    cache each analysis reloaded ~90MB of weights from disk. The cache lives at
    module level, and modules survive Streamlit re-runs — so the second run is
    instant. Works for the CLI too, unlike @st.cache_resource.
    """
    return SentenceTransformer(name)


class Embedder:
    def __init__(self, model="basic", llm_model=None, model_name=EMBEDDING_MODEL):
        self.model = model
        self.llm_model = llm_model
        self.model_name = model_name

    @classmethod
    def for_documents(cls, *texts):
        """Pick the encoder that can actually read these documents.

        all-MiniLM-L6-v2 is English-only. Handed an Arabic requirement it does
        not fail — it returns a vector, and that vector is noise, so the
        matcher goes on to compare noise against a resume and reports the
        result with the same confidence it reports everything else. Silent
        wrongness, which is the failure mode this project keeps running into
        and keeps having to design out.

        So the encoder is chosen by looking at the text. The cost of being
        wrong in the cheap direction is a larger download; the cost of being
        wrong in the other direction is a meaningless answer presented as a
        real one.
        """
        if MULTILINGUAL_MODEL and any(has_arabic(t) for t in texts if t):
            return cls(model_name=MULTILINGUAL_MODEL)
        return cls()

    def dim(self):
        """The loaded model's output width.

        Read from the model rather than config.yaml so that swapping the
        embedding model in a comparison run cannot silently produce vectors of
        one size and zero-fallbacks of another.
        """
        try:
            model = load_model(self.model_name)
            # sentence-transformers 5.x renamed this; support both so a
            # library upgrade doesn't silently fall back to the config value.
            getter = (getattr(model, "get_embedding_dimension", None)
                      or model.get_sentence_embedding_dimension)
            return int(getter())
        except Exception:
            return EMBEDDING_DIM

    def get_embeddings(self, text, model=None):
        """Turn text into a vector. `model` overrides the backend for one call."""
        backend = model or self.model
        if backend == "basic":
            return self.embed_basic(text)
        elif backend == "ollama":
            return self.embed_ollama(text)
        elif backend == "openai":
            return self.embed_openai(text)
        else:
            raise ValueError(f"Unknown model type: {backend}")

    def embed_basic(self, text):
        """Embed a document of any length.

        all-MiniLM-L6-v2 accepts 256 word-piece tokens and silently drops
        everything past that — no warning, no error. A typical resume is ~630
        tokens, so roughly 60% of it was never being read, and the similarity
        score described only the opening of each document.

        So: split into overlapping chunks that fit the window, embed each, and
        take a length-weighted mean. Averaging does blur a document that covers
        several unrelated topics, but reading 100% of it blurred is strictly
        better than reading 40% of it sharply.
        """
        if not text or not text.strip():
            return np.zeros(self.dim())

        model = load_model(self.model_name)
        chunks = self._chunk(text)

        if len(chunks) == 1:
            return model.encode(chunks[0])

        vectors = model.encode(chunks)
        weights = np.array([len(c.split()) for c in chunks], dtype=float)
        return np.average(vectors, axis=0, weights=weights)

    def embed_many(self, texts):
        """Embed a list of texts in one batched pass.

        Requirement-level matching embeds dozens of short strings at once;
        encoding them one call at a time wastes most of the GPU/CPU batch.
        Chunking still applies per text, so a long entry is handled the same
        way it would be on its own.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim()))

        model = load_model(self.model_name)

        # Flatten every text into its chunks, remembering which text each came
        # from, so one encode() call covers the whole batch.
        flat, owner = [], []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            for chunk in self._chunk(text):
                flat.append(chunk)
                owner.append(i)

        out = np.zeros((len(texts), self.dim()))
        if not flat:
            return out

        vectors = np.asarray(model.encode(flat))
        for i in range(len(texts)):
            rows = [v for v, o in zip(vectors, owner) if o == i]
            if rows:
                weights = [len(flat[j].split()) for j, o in enumerate(owner) if o == i]
                out[i] = np.average(rows, axis=0, weights=weights)
        return out

    def _window(self):
        """How many words fit the loaded model's token window, with headroom.

        This used to be hardcoded at 150 words, which is right for
        all-MiniLM-L6-v2's 256 word-pieces and wrong for anything else. The
        multilingual model this project evaluates accepts 128 — half as much —
        so a fixed 150 would have silently truncated every chunk and the
        Arabic results would have been quietly meaningless.

        That is the same bug the chunking work fixed in the first place, which
        is the argument for deriving the number instead of writing it down: a
        constant that is only correct for one model will outlive that model.

        The 0.55 words-per-token ratio is deliberately pessimistic. English
        prose runs nearer 0.75, but Arabic script and rare technical tokens
        fragment much harder, and over-filling the window loses text with no
        warning while under-filling it only costs a little sharpness.
        """
        try:
            limit = int(load_model(self.model_name).max_seq_length)
        except Exception:
            limit = 256
        # Two special tokens, then a safety margin.
        return max(32, int((limit - 2) * 0.55))

    def _chunk(self, text, words_per_chunk=None, overlap=None):
        """Split into word windows that comfortably fit the token limit.

        The overlap keeps a requirement that straddles a boundary from being
        split across two chunks and diluted in both.
        """
        if words_per_chunk is None:
            words_per_chunk = self._window()
        if overlap is None:
            overlap = max(4, words_per_chunk // 6)

        words = text.split()
        if len(words) <= words_per_chunk:
            return [text]

        step = words_per_chunk - overlap
        chunks = []
        for start in range(0, len(words), step):
            chunk = words[start:start + words_per_chunk]
            if not chunk:
                break
            chunks.append(" ".join(chunk))
            if start + words_per_chunk >= len(words):
                break
        return chunks

    def embed_ollama(self, text):
        raise NotImplementedError("Ollama embeddings are not wired up yet.")

    def embed_openai(self, text):
        raise NotImplementedError("OpenAI embeddings are not wired up yet.")
