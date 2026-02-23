"""
E-T-A RAG Prototype - SAP Data Connector

From Architecture Design (Section 5.2):
- SAP is NOT queried live
- Scheduled exports (nightly/weekly) as CSV or JSON
- Records serialized into natural language sentences

Two approaches:
1. Scheduled Export + Embed (descriptive queries)
2. Text-to-SQL (analytical queries) - handled in rag_engine.py
"""
import hashlib
import os
from typing import Optional

import pandas as pd

from connectors.base import BaseConnector, DocumentChunk


class SAPConnector(BaseConnector):
    """
    Connector for SAP data exports (CSV/JSON).
    
    Serializes SAP records into natural language for embedding.
    Example: Material CB-2410 → "Material CB-2410 is a Thermal Circuit 
    Breaker, measured in pieces, classified under ELEC-CB."
    """

    # SAP module → natural language templates
    TEMPLATES = {
        "material_master": (
            "Material {material_id} ({description}) is a {material_type}, "
            "measured in {unit}, classified under {classification}. "
            "Weight: {weight} {weight_unit}."
        ),
        "customer_master": (
            "Customer {customer_id} ({name}) is located in {city}, {country}. "
            "Contact: {contact_person}. Account group: {account_group}."
        ),
        "inventory": (
            "Material {material_id} has {stock_qty} units in stock "
            "at plant {plant}. Last updated: {last_updated}."
        ),
        "production_order": (
            "Production order {order_id} for material {material_id}: "
            "quantity {order_qty}, status {status}, "
            "planned start {start_date}, planned end {end_date}."
        ),
        "financial_posting": (
            "FI posting {doc_number}: {description}, amount {amount} {currency}, "
            "cost center {cost_center}, posting date {posting_date}."
        ),
    }

    def __init__(self, sap_module: str = "material_master"):
        self.sap_module = sap_module

    def extract(self, file_path: str, metadata: Optional[dict] = None) -> list[DocumentChunk]:
        """Extract SAP records from CSV export and serialize to natural language."""
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        chunks = []

        template = self.TEMPLATES.get(self.sap_module)

        for idx, row in df.iterrows():
            row_dict = {k: str(v) if pd.notna(v) else "N/A" for k, v in row.items()}

            if template:
                try:
                    text = template.format(**row_dict)
                except KeyError:
                    # Fallback: serialize as key-value pairs
                    text = self._row_to_text(row_dict)
            else:
                text = self._row_to_text(row_dict)

            text = f"[SAP {self.sap_module.replace('_', ' ').title()} Export]\n{text}"

            chunk_id = hashlib.md5(
                f"{file_path}:{self.sap_module}:{idx}".encode()
            ).hexdigest()

            chunk = DocumentChunk(
                text=text,
                source_format="sap_csv",
                source_file=filename,
                doc_type="sap_export",
                page_number=idx,
                chunk_id=chunk_id,
            )
            self._apply_metadata(chunk, metadata)
            chunks.append(chunk)

        return chunks

    def _row_to_text(self, row_dict: dict) -> str:
        """Fallback: serialize row as natural language key-value pairs."""
        parts = [f"{k}: {v}" for k, v in row_dict.items() if v != "N/A"]
        return "; ".join(parts) + "."
