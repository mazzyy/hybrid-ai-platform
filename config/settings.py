"""
E-T-A RAG Prototype - Configuration
Maps to production architecture but uses lightweight local components.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / ".vectorstore"

# ─── Embedding Model ────────────────────────────────────────────────
# Production: BGE-M3 (1024-dim, ~2GB VRAM)
# Prototype:  all-MiniLM-L6-v2 (384-dim, ~80MB, CPU-only)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # Changes to 1024 for BGE-M3

# ─── LLM Configuration ──────────────────────────────────────────────
# Production: On-prem vLLM (Llama 70B) or Cloud API
# Prototype:  Ollama local (Mistral 7B or Phi-3)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# ─── Vector Store ────────────────────────────────────────────────────
# Production: Qdrant (Docker, 16GB RAM, SSD storage)
# Prototype:  ChromaDB (embedded, persistent local storage)
CHROMA_COLLECTION = "eta_knowledge_base"

# ─── Chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens (matches production design)
CHUNK_OVERLAP = 50        # token overlap between chunks

# ─── Retrieval ───────────────────────────────────────────────────────
TOP_K_RESULTS = 5         # Number of chunks to retrieve
HYBRID_ALPHA = 0.7        # Weight: 0=pure BM25, 1=pure vector
RRF_K = 60                # Reciprocal Rank Fusion constant

# ─── Text-to-SQL (for structured/SAP data) ───────────────────────────
# Production: Read-only PostgreSQL staging DB
# Prototype:  SQLite in-memory
SQL_DB_PATH = PROJECT_ROOT / ".staging.db"

# ─── API ─────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Department Knowledge Spaces ─────────────────────────────────────
DEPARTMENTS = [
    "engineering",
    "production",
    "quality",
    "sales",
    "finance",
    "data_analytics",
    "hr",
    "it",
]

# ─── Metadata Schema (Layer 2 - Unified across all sources) ─────────
METADATA_FIELDS = {
    "department": str,      # Which department owns this data
    "doc_type": str,        # e.g., "spec", "policy", "report", "sap_export"
    "source_format": str,   # e.g., "pdf", "xlsx", "csv", "docx"
    "access_level": str,    # e.g., "all_staff", "department", "management"
    "date": str,            # ISO date of document
    "owner": str,           # Data owner / author
    "source_file": str,     # Original filename for provenance
    "page_number": int,     # Page/sheet/row reference
}
