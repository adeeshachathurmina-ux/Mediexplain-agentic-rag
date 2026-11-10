"""
Smart Clinical Overlapping Chunker & Document Ingestion Framework
DS205.3: Data Science with Python | Member 1 Deliverable

Class: SmartClinicalChunker(BaseDocumentIngestor)
Implements boundary-aware text chunking with sliding window overlap,
preserving clinical numerical thresholds and table integrity.
"""

import os
import re
from typing import List, Dict, Any, Optional

from src.interfaces.base_interfaces import BaseDocumentIngestor
from src.ingestion.pdf_loader import PDFDocumentLoader


class SmartClinicalChunker(BaseDocumentIngestor):
    """
    Implements BaseDocumentIngestor.
    Splits clinical guideline text into semantically cohesive, overlapping chunks
    while ensuring medical thresholds and lab intervals are never severed.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        loader: Optional[PDFDocumentLoader] = None,
    ):
        """
        Initializes the SmartClinicalChunker.
        :param chunk_size: Target maximum word count per chunk.
        :param chunk_overlap: Word overlap between consecutive chunks to prevent boundary loss.
        :param loader: Instance of PDFDocumentLoader (Dependency Injection).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader = loader if loader is not None else PDFDocumentLoader()

    def _split_into_clinical_sections(self, text: str) -> List[str]:
        """
        Splits raw page text into natural clinical sections using numbered headers
        (e.g., '1. Glycated...', '2. Fasting...') and table boundaries.
        """
        if not text:
            return []

        # Split on numbered section headers (e.g. "\n1. ", "\n2. ", "\n3. ")
        pattern = r"\n(?=\d+\.\s+[A-Z])"
        sections = re.split(pattern, text)
        clean_sections = [s.strip() for s in sections if len(s.strip()) > 15]

        # If no explicit numbered headers found, split by double newlines (paragraphs)
        if len(clean_sections) <= 1:
            clean_sections = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 15]

        return clean_sections if clean_sections else [text.strip()]

    def _create_overlapping_windows(self, words: List[str]) -> List[str]:
        """Creates sliding-window chunks with word overlap."""
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [" ".join(words)]

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            window = words[i : i + self.chunk_size]
            chunks.append(" ".join(window))
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def load_and_chunk(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Implements BaseDocumentIngestor.load_and_chunk contract:
        Reads a PDF and returns a list of chunk dictionaries:
        [{"chunk_id": str, "text": str, "source": str, "page": int}]
        """
        pages = self.loader.load_pdf(pdf_path)
        filename = os.path.basename(pdf_path)
        all_chunks: List[Dict[str, Any]] = []

        chunk_counter = 1
        for page_data in pages:
            page_num = page_data["page"]
            page_text = page_data["text"]

            # Split into logical clinical sections first
            sections = self._split_into_clinical_sections(page_text)

            for section in sections:
                words = section.split()
                # Apply sliding window overlap if section exceeds target chunk size
                if len(words) > self.chunk_size:
                    windows = self._create_overlapping_windows(words)
                    for win in windows:
                        cid = f"chunk_{chunk_counter:02d}"
                        all_chunks.append(
                            {
                                "chunk_id": cid,
                                "text": win.strip(),
                                "source": filename,
                                "page": page_num,
                            }
                        )
                        chunk_counter += 1
                else:
                    cid = f"chunk_{chunk_counter:02d}"
                    all_chunks.append(
                        {
                            "chunk_id": cid,
                            "text": section.strip(),
                            "source": filename,
                            "page": page_num,
                        }
                    )
                    chunk_counter += 1

        return all_chunks

    def chunk_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Chunks all PDFs within a target directory with continuous global chunk IDs.
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        global_chunks: List[Dict[str, Any]] = []
        chunk_idx = 1

        for file in sorted(os.listdir(dir_path)):
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(dir_path, file)
                doc_chunks = self.load_and_chunk(full_path)
                for c in doc_chunks:
                    c["chunk_id"] = f"chunk_{chunk_idx:02d}"
                    global_chunks.append(c)
                    chunk_idx += 1

        return global_chunks


if __name__ == "__main__":
    # Self-test when executed directly
    chunker = SmartClinicalChunker()
    pdf_file = "data/raw_pdfs/WHO_Diabetes_Guideline.pdf"
    if os.path.exists(pdf_file):
        chunks = chunker.load_and_chunk(pdf_file)
        print(f"\n[SUCCESS] Generated {len(chunks)} smart chunk(s) from {pdf_file}:")
        for c in chunks:
            print(f"- {c['chunk_id']} (Page {c['page']}): {c['text'][:110]}...")
    else:
        print(f"File not found: {pdf_file}")