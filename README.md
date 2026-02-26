# E-T-A RAG System - MacBook Prototype

## Architecture Mapping: Production → Prototype

This prototype mirrors the full E-T-A RAG architecture design (v2.0) using lightweight components that run entirely on a MacBook (Apple Silicon or Intel, 16GB+ RAM recommended).

| Production Component | Prototype Replacement | Why |
|---|---|---|
| BGE-M3 (2GB VRAM, GPU) | `all-MiniLM-L6-v2` (80MB, CPU) | Fast, good quality, runs on CPU |
| Qdrant (Docker, 16GB RAM) | Qdrant in-memory or ChromaDB | Zero-config, embedded mode |
| On-prem LLM (e.g. Llama 70B) | `Ollama + Mistral 7B` or `Phi-3 mini` | Runs on MacBook with 8GB+ RAM |
| LangChain orchestration | LangChain (same) | Same framework, same patterns |
| Docker containers | Local Python processes | No Docker needed for prototype |
| SAP RFC/BAPI export | Sample CSV files | Simulated SAP exports |
| SSO + RBAC | Simulated user roles | Same filtering logic, mock auth |

## Architecture Layers (Same as Production Design)

```
┌─────────────────────────────────────────────────┐
│  User Query → [Mock Auth] → [Query Router]      │
└───────────────────────┬─────────────────────────┘
                        │
              Needs internal knowledge?
           No │                    Yes │
              │                        ▼
              │     ┌───────────────────────────┐
              │     │   RAG RETRIEVAL ENGINE     │
              │     │   [RBAC Filter]            │
              │     │   [Vector + BM25 Search]   │
              │     └─────────────┬─────────────┘
              │                   │ context
              ▼                   ▼
┌─────────────────────────────────────────────────┐
│   Ollama (Mistral 7B / Phi-3) Local LLM         │
└─────────────────────────────────────────────────┘
```

## Three-Layer Data Organization (Same as Production)

```
Layer 1: Format-Specific Connectors
  → Excel Parser, PDF Extractor, SAP CSV Connector, DOCX Parser

Layer 2: Unified Metadata Tagging
  → dept | doc_type | source | access_level | date | owner

Layer 3: Department Knowledge Spaces
  → [Engineering] [HR] [Finance] [Quality]
```

## Quick Start

### 1. Prerequisites

```bash
# Install Ollama (local LLM runtime)
brew install ollama

# Pull a small model (pick one)
ollama pull mistral        # 4.1GB - good quality, needs 8GB+ RAM
ollama pull phi3:mini       # 2.3GB - lighter, works on 8GB RAM

# Create Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest Sample Data

```bash
# This ingests all sample documents from data/ into the vector store
python scripts/ingest.py
```

### 3. Run the RAG API

```bash
# Start the FastAPI server
python api/server.py

# Open http://localhost:8000/docs for the Swagger UI
```

### 4. Query via CLI

```bash
# Interactive chat with RAG
python scripts/chat.py

# Single query
python scripts/chat.py --query "What is the thermal rating of CB-2410?"
```

### 5. Query via Web UI

To use the web UI, you need to ensure Ollama is running in the background first so that the local LLM is accessible.

```bash
# 1. Start Ollama (if not already running via the Mac app)
# You can start it from your Applications folder, or run:
ollama serve

# 2. In a NEW terminal window, start the Streamlit UI
streamlit run api/ui.py
```

## Sample Data Included

The `data/` folder contains simulated E-T-A documents:

- `engineering/` - Product specs (PDF-like), test reports (Excel-like)
- `hr/` - Employee handbook, policies
- `finance/` - Budget reports, SAP export samples
- `quality/` - Audit checklists, certification docs

## RBAC Simulation

The prototype simulates the same RBAC matrix from the architecture design:

| Data Category | all_staff | engineering | hr | management |
|---|---|---|---|---|
| Product Manuals | ✓ | ✓ | ✓ | ✓ |
| Engineering Specs | ✗ | ✓ | ✗ | ✓ |
| HR Records | ✗ | ✗ | ✓ | ✓ |
| Financial Reports | ✗ | ✗ | ✗ | ✓ |

## Project Structure

```
eta-rag-prototype/
├── README.md
├── requirements.txt
├── config/
│   ├── settings.py          # All configuration
│   └── rbac.py              # Role-based access control rules
├── connectors/
│   ├── base.py              # Base connector interface
│   ├── excel_connector.py   # Excel/CSV ingestion
│   ├── pdf_connector.py     # PDF text extraction
│   ├── docx_connector.py    # Word document parsing
│   └── sap_connector.py     # SAP CSV export connector
├── retrieval/
│   ├── embedder.py          # Embedding service
│   ├── vector_store.py      # ChromaDB/Qdrant wrapper
│   ├── hybrid_search.py     # Vector + BM25 hybrid search
│   └── rag_engine.py        # Main RAG orchestration
├── api/
│   ├── server.py            # FastAPI REST API
│   └── ui.py                # Streamlit web UI
├── scripts/
│   ├── ingest.py            # Batch ingestion script
│   └── chat.py              # CLI chat interface
└── data/                    # Sample documents
    ├── engineering/
    ├── hr/
    ├── finance/
    └── quality/
```

## Scaling to Production

When moving from prototype to production:

1. Replace `all-MiniLM-L6-v2` → `BGE-M3` (just change config)
2. Replace ChromaDB → Qdrant Docker (change vector_store.py backend)
3. Replace Ollama → on-prem vLLM or cloud API (change LLM endpoint)
4. Add real SSO → replace mock auth with AD integration
5. Add real SAP exports → replace sample CSVs with RFC/BAPI pipeline
6. Containerize with Docker Compose
