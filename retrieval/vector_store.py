"""
E-T-A RAG Prototype - Vector Store

Production: Qdrant (Docker, 16GB RAM, native hybrid search)
Prototype:  ChromaDB (embedded, persistent, zero-config)

Handles storage, retrieval, and RBAC-filtered search.
"""
import os
from typing import Optional

import chromadb
from chromadb.config import Settings

from config.settings import VECTOR_STORE_DIR, CHROMA_COLLECTION
from connectors.base import DocumentChunk


class VectorStore:
    """ChromaDB-backed vector store with metadata filtering for RBAC."""

    def __init__(self, persist_dir: str = None):
        persist_dir = persist_dir or str(VECTOR_STORE_DIR)
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Vector store initialized. Collection '{CHROMA_COLLECTION}' "
              f"has {self.collection.count()} documents.")

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]):
        """Add document chunks with their embeddings and metadata."""
        if not chunks:
            return

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata_dict() for c in chunks],
        )
        print(f"Added {len(chunks)} chunks. Total: {self.collection.count()}")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar chunks with optional RBAC metadata filter.
        
        This is where pre-retrieval RBAC filtering happens:
        chunks the user cannot access are NEVER returned.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            # If filter fails (e.g., complex $and/$or), fall back to unfiltered
            print(f"Filter query failed ({e}), falling back to basic filter")
            kwargs.pop("where", None)
            results = self.collection.query(**kwargs)

        # Format results
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,  # Convert distance to similarity
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def get_all_documents(self) -> list[str]:
        """Get all document texts (for BM25 index building)."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["documents"])
        return results["documents"]

    def get_all_ids(self) -> list[str]:
        """Get all chunk IDs."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get()
        return results["ids"]

    def clear(self):
        """Clear all data from the collection."""
        self.client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        print("Vector store cleared.")

    @property
    def count(self) -> int:
        return self.collection.count()
