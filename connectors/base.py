"""
E-T-A RAG Prototype - Base Connector Interface
Layer 1: Format-Specific Connectors

Every connector extracts content from a specific format and outputs
standardized chunks with unified metadata (Layer 2).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentChunk:
    """
    A single chunk ready for embedding and storage.
    Carries the unified metadata schema from Layer 2.
    """
    text: str                           # The actual content
    # Layer 2: Unified Metadata
    department: str = ""                # e.g., "engineering", "hr"
    doc_type: str = ""                  # e.g., "spec", "policy", "report"
    source_format: str = ""             # e.g., "pdf", "xlsx", "csv", "docx"
    access_level: str = "public"        # "public", "department", "restricted", "confidential"
    date: str = ""                      # ISO date
    owner: str = ""                     # Author / data owner
    source_file: str = ""              # Original filename (provenance)
    page_number: int = 0                # Page, sheet, or row reference
    chunk_id: str = ""                  # Unique identifier
    extra_metadata: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        """Return metadata as a flat dict for vector store storage."""
        meta = {
            "department": self.department,
            "doc_type": self.doc_type,
            "source_format": self.source_format,
            "access_level": self.access_level,
            "date": self.date,
            "owner": self.owner,
            "source_file": self.source_file,
            "page_number": self.page_number,
        }
        meta.update(self.extra_metadata)
        return meta


class BaseConnector(ABC):
    """
    Abstract base for all format-specific connectors.
    Each connector knows how to:
    1. Read files of its format
    2. Extract text content
    3. Chunk the content appropriately
    4. Attach unified metadata (Layer 2)
    """

    @abstractmethod
    def extract(self, file_path: str, metadata: Optional[dict] = None) -> list[DocumentChunk]:
        """
        Extract and chunk content from a file.
        
        Args:
            file_path: Path to the source file
            metadata: Optional override metadata (department, access_level, etc.)
        
        Returns:
            List of DocumentChunk objects ready for embedding
        """
        pass

    def _apply_metadata(self, chunk: DocumentChunk, metadata: Optional[dict] = None):
        """Apply override metadata to a chunk."""
        if metadata:
            for key, value in metadata.items():
                if hasattr(chunk, key):
                    setattr(chunk, key, value)
        return chunk
