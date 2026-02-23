#!/usr/bin/env python3
"""
E-T-A RAG Prototype - Batch Ingestion Script

Ingests all sample documents from data/ into the vector store,
applying appropriate metadata (Layer 2) for each department.

Usage:
    python scripts/ingest.py [--clear]
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from connectors import ExcelConnector, SAPConnector, DocumentChunk
from connectors.base import BaseConnector
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from config.settings import DATA_DIR
import hashlib


class TextConnector(BaseConnector):
    """Simple connector for .txt files (simulating PDF/DOCX in prototype)."""
    def extract(self, file_path, metadata=None):
        with open(file_path, "r") as f:
            text = f.read()
        filename = os.path.basename(file_path)
        
        # Split into chunks of ~2000 chars
        chunks = []
        parts = text.split("\n\n")
        current = ""
        chunk_idx = 0
        
        for part in parts:
            if len(current) + len(part) > 2000 and current:
                chunk_id = hashlib.md5(f"{file_path}:{chunk_idx}".encode()).hexdigest()
                chunk = DocumentChunk(
                    text=f"[Source: {filename}]\n{current.strip()}",
                    source_format="txt",
                    source_file=filename,
                    page_number=chunk_idx,
                    chunk_id=chunk_id,
                )
                self._apply_metadata(chunk, metadata)
                chunks.append(chunk)
                current = part
                chunk_idx += 1
            else:
                current = f"{current}\n\n{part}" if current else part
        
        if current.strip():
            chunk_id = hashlib.md5(f"{file_path}:{chunk_idx}".encode()).hexdigest()
            chunk = DocumentChunk(
                text=f"[Source: {filename}]\n{current.strip()}",
                source_format="txt",
                source_file=filename,
                page_number=chunk_idx,
                chunk_id=chunk_id,
            )
            self._apply_metadata(chunk, metadata)
            chunks.append(chunk)
        
        return chunks


# ─── Data Source Definitions ─────────────────────────────────────────
# Maps files to their metadata (Layer 2: Unified Metadata Tagging)
DATA_SOURCES = [
    # Engineering Department
    {
        "file": "engineering/cb2410_product_spec.txt",
        "connector": "text",
        "metadata": {
            "department": "engineering",
            "doc_type": "spec",
            "access_level": "department",
            "owner": "Thomas Mueller",
            "date": "2025-09-15",
        },
    },
    {
        "file": "engineering/test_reports.csv",
        "connector": "excel",
        "metadata": {
            "department": "engineering",
            "doc_type": "test_report",
            "access_level": "department",
            "owner": "Engineering Lab",
            "date": "2025-08-01",
        },
    },
    {
        "file": "engineering/sap_material_master.csv",
        "connector": "sap",
        "sap_module": "material_master",
        "metadata": {
            "department": "engineering",
            "doc_type": "sap_export",
            "access_level": "department",
            "owner": "SAP System",
            "date": "2025-12-01",
        },
    },
    # HR Department
    {
        "file": "hr/employee_handbook.txt",
        "connector": "text",
        "metadata": {
            "department": "hr",
            "doc_type": "policy",
            "access_level": "restricted",
            "owner": "Anna Weber",
            "date": "2025-01-15",
        },
    },
    # Finance Department
    {
        "file": "finance/budget_report_2025.csv",
        "connector": "excel",
        "metadata": {
            "department": "finance",
            "doc_type": "report",
            "access_level": "confidential",
            "owner": "Finance Team",
            "date": "2025-12-01",
        },
    },
    # Quality Department
    {
        "file": "quality/certifications_and_audits.txt",
        "connector": "text",
        "metadata": {
            "department": "quality",
            "doc_type": "certification",
            "access_level": "department",
            "owner": "Stefan Huber",
            "date": "2025-10-01",
        },
    },
]


def main():
    parser = argparse.ArgumentParser(description="Ingest E-T-A documents into RAG vector store")
    parser.add_argument("--clear", action="store_true", help="Clear vector store before ingesting")
    args = parser.parse_args()

    print("=" * 60)
    print("E-T-A RAG Prototype - Document Ingestion")
    print("=" * 60)

    # Initialize components
    embedder = Embedder()
    vector_store = VectorStore()

    if args.clear:
        print("\nClearing existing vector store...")
        vector_store.clear()

    # Initialize connectors
    connectors = {
        "text": TextConnector(),
        "excel": ExcelConnector(),
    }

    total_chunks = 0

    for source in DATA_SOURCES:
        file_path = DATA_DIR / source["file"]

        if not file_path.exists():
            print(f"\n⚠ Skipping {source['file']} (file not found)")
            continue

        print(f"\n📄 Processing: {source['file']}")
        print(f"   Department: {source['metadata']['department']}")
        print(f"   Access Level: {source['metadata']['access_level']}")

        # Select connector
        if source["connector"] == "sap":
            connector = SAPConnector(sap_module=source.get("sap_module", "material_master"))
        else:
            connector = connectors[source["connector"]]

        # Extract chunks
        chunks = connector.extract(str(file_path), metadata=source["metadata"])
        print(f"   Chunks extracted: {len(chunks)}")

        if chunks:
            # Embed and store
            texts = [c.text for c in chunks]
            embeddings = embedder.embed(texts)
            vector_store.add_chunks(chunks, embeddings)
            total_chunks += len(chunks)

    print(f"\n{'=' * 60}")
    print(f"✅ Ingestion complete!")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Vector store size: {vector_store.count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
