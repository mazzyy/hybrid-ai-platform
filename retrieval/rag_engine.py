"""
E-T-A RAG Prototype - RAG Engine

Main orchestration layer that mirrors the production architecture:
1. Query Router: Decides if RAG is needed
2. RBAC Filter: Applies user role-based access control
3. Hybrid Retrieval: Vector + BM25 search
4. LLM Generation: Ollama local model with retrieved context
5. Data Provenance: Source attribution in every response
"""
import json
from typing import Optional

import ollama

from config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL, TOP_K_RESULTS
from config.rbac import get_retrieval_filter, ROLE_ACCESS, MOCK_USERS
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.hybrid_search import HybridSearch


class RAGEngine:
    """
    Main RAG engine following the architecture design.
    Handles: Query routing → RBAC filtering → Retrieval → LLM generation
    """

    def __init__(self):
        print("Initializing RAG Engine...")
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.hybrid_search = HybridSearch(self.embedder, self.vector_store)
        self.model = OLLAMA_MODEL
        print(f"RAG Engine ready. LLM: {self.model} via Ollama")

    def query(
        self,
        question: str,
        user_role: str = "all_staff",
        user_name: str = "Anonymous",
        top_k: int = TOP_K_RESULTS,
    ) -> dict:
        """
        Full RAG pipeline:
        1. Build RBAC filter from user role
        2. Retrieve relevant chunks (hybrid search with RBAC)
        3. Build prompt with context
        4. Generate answer via local LLM
        5. Return answer with source provenance
        """
        # Step 1: RBAC filter
        rbac_filter = self._build_simple_filter(user_role)

        # Step 2: Hybrid retrieval with RBAC filtering
        results = self.hybrid_search.search(
            query=question,
            top_k=top_k,
            where_filter=rbac_filter,
        )

        if not results:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base that you have access to.",
                "sources": [],
                "user_role": user_role,
                "chunks_retrieved": 0,
            }

        # Step 3: Build prompt with context and provenance
        context = self._build_context(results)
        prompt = self._build_prompt(question, context, user_name)

        # Step 4: Generate answer via Ollama
        answer = self._generate(prompt)

        # Step 5: Extract source provenance
        sources = self._extract_sources(results)

        return {
            "answer": answer,
            "sources": sources,
            "user_role": user_role,
            "chunks_retrieved": len(results),
            "context_preview": context[:500] + "..." if len(context) > 500 else context,
        }

    def _build_simple_filter(self, user_role: str) -> Optional[dict]:
        """Build a ChromaDB-compatible filter based on user role."""
        role_config = ROLE_ACCESS.get(user_role, ROLE_ACCESS["all_staff"])
        allowed_levels = role_config["allowed_access_levels"]

        # Simple filter: just filter by access_level
        if len(allowed_levels) == 1:
            return {"access_level": {"$eq": allowed_levels[0]}}
        else:
            return {"access_level": {"$in": allowed_levels}}

    def _build_context(self, results: list[dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            source = meta.get("source_file", "Unknown")
            dept = meta.get("department", "Unknown")

            context_parts.append(
                f"--- Document {i} [Source: {source}, Department: {dept}] ---\n"
                f"{result['text']}\n"
            )
        return "\n".join(context_parts)

    def _build_prompt(self, question: str, context: str, user_name: str) -> str:
        """
        Build the LLM prompt with retrieved context.
        
        From Architecture Design (Section 8 - Security):
        "Retrieved chunks wrapped in structured delimiters, 
        treated as data not instructions."
        """
        return f"""You are an AI assistant for E-T-A Elektrotechnische Apparate GmbH.
Answer the employee's question using ONLY the provided context documents.
If the context doesn't contain enough information, say so clearly.
Always cite which source document(s) your answer comes from.

<context>
{context}
</context>

Employee ({user_name}): {question}

Answer:"""

    def _generate(self, prompt: str) -> str:
        """Generate answer using Ollama local LLM."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 1024},
            )
            return response["message"]["content"]
        except Exception as e:
            return (
                f"Error connecting to Ollama ({self.model}): {e}\n\n"
                "Make sure Ollama is running: `ollama serve`\n"
                f"And the model is pulled: `ollama pull {self.model}`"
            )

    def _extract_sources(self, results: list[dict]) -> list[dict]:
        """Extract source provenance from results."""
        sources = []
        seen = set()
        for result in results:
            meta = result.get("metadata", {})
            source_file = meta.get("source_file", "Unknown")
            if source_file not in seen:
                seen.add(source_file)
                sources.append({
                    "file": source_file,
                    "department": meta.get("department", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "page": meta.get("page_number", ""),
                })
        return sources

    def ingest_chunks(self, chunks: list) -> int:
        """Ingest document chunks into the vector store."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add_chunks(chunks, embeddings)

        # Rebuild BM25 index
        self.hybrid_search.rebuild_bm25()

        return len(chunks)
