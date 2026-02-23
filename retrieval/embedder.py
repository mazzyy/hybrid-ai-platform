"""
E-T-A RAG Prototype - Embedding Service

Production: BGE-M3 (1024-dim, multilingual DE/EN, ~2 GB VRAM)
Prototype:  all-MiniLM-L6-v2 (384-dim, ~80MB, CPU-only)

To switch to production model, change EMBEDDING_MODEL in config/settings.py
"""
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL


class Embedder:
    """Thin wrapper around sentence-transformers for embedding text."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of float vectors."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.model.encode(query).tolist()
