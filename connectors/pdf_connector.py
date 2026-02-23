"""
E-T-A RAG Prototype - PDF Connector

From Architecture Design (Section 5.1):
- Extraction: PyMuPDF / pdfplumber
- Chunking: Recursive splitting by sections, 512-token chunks
- Output: Text with page numbers
"""
import hashlib
import os
from pathlib import Path
from typing import Optional

from connectors.base import BaseConnector, DocumentChunk


class PDFConnector(BaseConnector):
    """Connector for PDF files using PyMuPDF."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract(self, file_path: str, metadata: Optional[dict] = None) -> list[DocumentChunk]:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        chunks = []
        filename = os.path.basename(file_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()

            if not text:
                continue

            # Split page text into chunks (~512 tokens ≈ ~2048 chars)
            page_chunks = self._split_text(text, max_chars=2048, overlap_chars=200)

            for i, chunk_text in enumerate(page_chunks):
                chunk_id = hashlib.md5(
                    f"{file_path}:p{page_num}:c{i}".encode()
                ).hexdigest()

                chunk = DocumentChunk(
                    text=f"[Source: {filename}, Page {page_num + 1}]\n{chunk_text}",
                    source_format="pdf",
                    source_file=filename,
                    page_number=page_num + 1,
                    chunk_id=chunk_id,
                )
                self._apply_metadata(chunk, metadata)
                chunks.append(chunk)

        doc.close()
        return chunks

    def _split_text(self, text: str, max_chars: int = 2048, overlap_chars: int = 200) -> list[str]:
        """Recursive text splitting with overlap."""
        if len(text) <= max_chars:
            return [text]

        # Try to split at paragraph boundaries
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) > max_chars and current:
                chunks.append(current.strip())
                # Keep overlap
                current = current[-overlap_chars:] + "\n\n" + para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append(current.strip())

        return chunks
