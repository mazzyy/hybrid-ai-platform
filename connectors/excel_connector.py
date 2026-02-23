"""
E-T-A RAG Prototype - Excel Connector
Handles .xlsx and .csv files.

From Architecture Design (Section 5.1):
- Extraction: openpyxl / pandas
- Chunking: Row groups serialized into natural language sentences
- Output: Text per logical record

Special handling (Section 4.2 / 5.1):
- Each sheet processed independently
- Tables with headers: rows serialized into sentences
- Summary sheets: treated as unstructured text
"""
import os
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

from connectors.base import BaseConnector, DocumentChunk


class ExcelConnector(BaseConnector):
    """Connector for Excel (.xlsx) and CSV files."""

    def extract(self, file_path: str, metadata: Optional[dict] = None) -> list[DocumentChunk]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".csv":
            return self._process_csv(file_path, metadata)
        elif ext in (".xlsx", ".xls"):
            return self._process_excel(file_path, metadata)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _process_csv(self, file_path: str, metadata: Optional[dict] = None) -> list[DocumentChunk]:
        """Process a CSV file (e.g., SAP export)."""
        df = pd.read_csv(file_path)
        return self._dataframe_to_chunks(df, file_path, sheet_name="csv", metadata=metadata)

    def _process_excel(self, file_path: str, metadata: Optional[dict] = None) -> list[DocumentChunk]:
        """Process an Excel file, each sheet independently."""
        chunks = []
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            sheet_chunks = self._dataframe_to_chunks(
                df, file_path, sheet_name=sheet_name, metadata=metadata
            )
            chunks.extend(sheet_chunks)
        return chunks

    def _dataframe_to_chunks(
        self,
        df: pd.DataFrame,
        file_path: str,
        sheet_name: str = "",
        metadata: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        """
        Convert a DataFrame into natural language chunks.
        
        Strategy from architecture design:
        - Tables with clear headers → rows serialized into sentences
        - Each row becomes a natural language sentence
        """
        chunks = []
        filename = os.path.basename(file_path)
        columns = list(df.columns)

        # Group rows into chunks of ~5 rows for context
        group_size = 5
        for i in range(0, len(df), group_size):
            group = df.iloc[i:i + group_size]
            sentences = []

            for _, row in group.iterrows():
                # Serialize row into a natural language sentence
                parts = []
                for col in columns:
                    val = row[col]
                    if pd.notna(val):
                        parts.append(f"{col}: {val}")
                if parts:
                    sentences.append("; ".join(parts) + ".")

            if not sentences:
                continue

            text = f"[Source: {filename}, Sheet: {sheet_name}]\n" + "\n".join(sentences)

            chunk_id = hashlib.md5(
                f"{file_path}:{sheet_name}:{i}".encode()
            ).hexdigest()

            chunk = DocumentChunk(
                text=text,
                source_format="xlsx" if file_path.endswith((".xlsx", ".xls")) else "csv",
                source_file=filename,
                page_number=i,  # Row group index
                chunk_id=chunk_id,
            )
            self._apply_metadata(chunk, metadata)
            chunks.append(chunk)

        return chunks
