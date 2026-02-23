"""
E-T-A RAG Prototype - Hybrid Search Engine

From Architecture Design (Section 6):
- Dense vector similarity (semantic meaning)
- Sparse BM25 search (exact keyword matching)
- Results merged using Reciprocal Rank Fusion (RRF)

"This is important because E-T-A queries often mix natural language
with specific part numbers or codes."
"""
from typing import Optional

from rank_bm25 import BM25Okapi

from config.settings import TOP_K_RESULTS, HYBRID_ALPHA, RRF_K
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore


class HybridSearch:
    """
    Combines vector search (semantic) with BM25 (keyword) using
    Reciprocal Rank Fusion for best-of-both-worlds retrieval.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = None
        self.bm25_docs = None
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from all documents in the vector store."""
        docs = self.vector_store.get_all_documents()
        if docs:
            tokenized = [doc.lower().split() for doc in docs]
            self.bm25_index = BM25Okapi(tokenized)
            self.bm25_docs = docs
            print(f"BM25 index built with {len(docs)} documents.")
        else:
            print("No documents for BM25 index yet.")

    def rebuild_bm25(self):
        """Rebuild BM25 index (call after adding new documents)."""
        self._build_bm25_index()

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        where_filter: Optional[dict] = None,
        alpha: float = HYBRID_ALPHA,
    ) -> list[dict]:
        """
        Hybrid search combining vector similarity and BM25.
        
        Args:
            query: User's natural language query
            top_k: Number of results to return
            where_filter: RBAC metadata filter for vector search
            alpha: Weight between vector (1.0) and BM25 (0.0)
        
        Returns:
            List of dicts with text, metadata, and score
        """
        # 1. Vector search (semantic)
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Fetch more for fusion
            where_filter=where_filter,
        )

        # 2. BM25 search (keyword)
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        # 3. Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            vector_results, bm25_results, alpha=alpha
        )

        return fused[:top_k]

    def _bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 keyword search."""
        if not self.bm25_index or not self.bm25_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "text": self.bm25_docs[i],
                "metadata": {},
                "score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0
        ]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        alpha: float = 0.7,
        k: int = RRF_K,
    ) -> list[dict]:
        """
        Merge results using Reciprocal Rank Fusion.
        RRF score = alpha * (1 / (k + rank_vector)) + (1-alpha) * (1 / (k + rank_bm25))
        """
        # Build text → result mapping
        all_results = {}

        # Score vector results
        for rank, result in enumerate(vector_results):
            text = result["text"]
            rrf_score = alpha * (1.0 / (k + rank + 1))
            all_results[text] = {
                "text": text,
                "metadata": result.get("metadata", {}),
                "score": rrf_score,
                "vector_score": result.get("score", 0),
            }

        # Add BM25 scores
        for rank, result in enumerate(bm25_results):
            text = result["text"]
            bm25_rrf = (1 - alpha) * (1.0 / (k + rank + 1))
            if text in all_results:
                all_results[text]["score"] += bm25_rrf
                all_results[text]["bm25_score"] = result.get("score", 0)
            else:
                all_results[text] = {
                    "text": text,
                    "metadata": result.get("metadata", {}),
                    "score": bm25_rrf,
                    "bm25_score": result.get("score", 0),
                }

        # Sort by fused score
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results
