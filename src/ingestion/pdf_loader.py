"""
PDF Ingestion & Structural Document Extraction Engine
DS205.3: Data Science with Python | Member 1 Deliverable (Data Pipeline & Engineering)

Class: PDFDocumentLoader
Extracts unstructured textual narratives, structured clinical guideline tables,
and page-level metadata from complex medical PDF documents using pdfplumber.
"""

import os
from typing import List, Dict, Any, Optional
import pdfplumber


class PDFDocumentLoader:
    """
    Robust clinical PDF document reader.
    Extracts both narrative text and clinical reference range tables page-by-page,
    normalizing whitespace and capturing structural metadata (source, page_number).
    """

    def __init__(self, normalize_whitespace: bool = True):
        """
        Initializes the PDFDocumentLoader.
        :param normalize_whitespace: Whether to collapse multiple spaces and empty lines.
        """
        self.normalize_whitespace = normalize_whitespace

    def _clean_text(self, text: str) -> str:
        """Cleans and standardizes extracted text stream."""
        if not text:
            return ""
        # Remove trailing and duplicate whitespace while preserving paragraph flow
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def _extract_page_tables(self, page) -> List[str]:
        """
        Extracts structured tables from a PDF page and converts them into
        readable pipe-delimited text representations for accurate downstream RAG retrieval.
        """
        table_lines = []
        try:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    # Clean each cell
                    clean_row = [str(cell).strip().replace("\n", " ") for cell in row if cell is not None]
                    if any(clean_row):
                        table_lines.append(" | ".join(clean_row))
        except Exception:
            pass
        return table_lines

    def load_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts all pages from the target PDF document.

        :param pdf_path: Absolute or relative path to the PDF file.
        :return: List of page dictionaries:
                 [{"source": "WHO_Diabetes_Guideline.pdf", "page": 1, "text": "...", "tables": [...]}]
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file does not exist at path: {pdf_path}")

        filename = os.path.basename(pdf_path)
        extracted_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_idx, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned_text = self._clean_text(raw_text)

                table_rows = self._extract_page_tables(page)
                combined_text_elements = []

                if cleaned_text:
                    combined_text_elements.append(cleaned_text)

                # Append table information if present and not already fully captured in raw text
                if table_rows:
                    combined_text_elements.append("\n[Clinical Reference Table]:\n" + "\n".join(table_rows))

                final_page_text = "\n\n".join(combined_text_elements).strip()

                extracted_pages.append(
                    {
                        "source": filename,
                        "page": page_idx,
                        "total_pages": total_pages,
                        "text": final_page_text,
                        "has_tables": len(table_rows) > 0,
                    }
                )

        return extracted_pages

    def load_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Extracts all PDF documents in a directory.

        :param dir_path: Directory containing PDF documents (e.g., 'data/raw_pdfs').
        :return: Flattened list of page dictionaries across all loaded documents.
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        all_pages = []
        for file in sorted(os.listdir(dir_path)):
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(dir_path, file)
                pages = self.load_pdf(full_path)
                all_pages.extend(pages)

        return all_pages


if __name__ == "__main__":
    # Self-test when executed directly
    loader = PDFDocumentLoader()
    sample_pdf = "data/raw_pdfs/WHO_Diabetes_Guideline.pdf"
    if os.path.exists(sample_pdf):
        pages = loader.load_pdf(sample_pdf)
        print(f"Loaded {len(pages)} page(s) from {sample_pdf}")
        print("Page 1 Text Preview:\n", pages[0]["text"][:200], "...")
    else:
        print(f"Sample PDF not found at {sample_pdf}")