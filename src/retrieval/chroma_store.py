"""
Persistent Vector Database & Semantic Embeddings Store (ChromaDB)
DS205.3: Data Science with Python | Member 1 Deliverable (Vector Persistence)

Class: ChromaVectorStore
Persistently stores and queries clinical guideline chunk embeddings in './data/chroma_db'
using sentence-transformers ('all-MiniLM-L6-v2') and ChromaDB PersistentClient.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import SmartClinicalChunker


class ChromaVectorStore:
    """
    Manages persistent on-disk embedding storage and nearest-neighbor vector search.
    Ensures vector index and document metadata survive across application restarts.
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "clinical_guidelines",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initializes the persistent ChromaDB client and dense embedding model.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize persistent ChromaDB storage client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Load embedding model
        self.encoder = SentenceTransformer(self.embedding_model_name)

        # Retrieve or create clinical collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "MediExplain Clinical Practice Guidelines Vectors"},
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Generates dense embeddings and upserts chunks into the persistent Chroma store.
        :param chunks: List of chunk dictionaries:
                       [{"chunk_id": str, "text": str, "source": str, "page": int}]
        :return: Number of chunks successfully stored.
        """
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "page": int(c["page"])} for c in chunks]

        # Generate 384-dimensional dense vectors
        embeddings = self.encoder.encode(documents, normalize_embeddings=True).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against stored clinical guideline embeddings.
        :param query_text: Patient or clinical question.
        :param top_k: Number of nearest chunks to retrieve.
        :return: Ranked list of chunks with similarity scores.
        """
        if not query_text or self.count() == 0:
            return []

        query_vector = self.encoder.encode([query_text], normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                # Convert L2 / cosine distance to similarity score
                similarity = round(max(0.0, 1.0 - (dist / 2.0)), 4)
                formatted_results.append(
                    {
                        "chunk_id": cid,
                        "text": doc,
                        "source": meta.get("source", "Unknown"),
                        "page": meta.get("page", 1),
                        "distance": round(dist, 4),
                        "similarity_score": similarity,
                    }
                )

        return formatted_results

    def count(self) -> int:
        """Returns the total number of chunks persistently stored in the collection."""
        return self.collection.count()

    def ingest_all_guidelines(self, pdf_dir: str = "data/raw_pdfs") -> int:
        """
        Convenience method: Reads all clinical PDFs, chunks them with SmartClinicalChunker,
        and persists them into ChromaDB.
        """
        chunker = SmartClinicalChunker()
        chunks = chunker.chunk_directory(pdf_dir)
        count_added = self.add_chunks(chunks)
        return count_added


if __name__ == "__main__":
    # Self-test when executed directly
    print("\n--- Initializing ChromaDB Persistent Vector Store ---")
    store = ChromaVectorStore()

    pdf_dir = "data/raw_pdfs"
    if os.path.exists(pdf_dir):
        print(f"Ingesting all clinical guideline PDFs from '{pdf_dir}'...")
        total_ingested = store.ingest_all_guidelines(pdf_dir)
        print(f"[SUCCESS] Persistently stored {total_ingested} chunks in ChromaDB!")
        print(f"Total Vector Count in '{store.persist_directory}': {store.count()}")

        # Test query
        test_q = "What is the threshold for diabetes mellitus?"
        print(f"\nTesting Semantic Vector Query: '{test_q}'")
        matches = store.query(test_q, top_k=2)
        for m in matches:
            print(f"- [{m['chunk_id']}] Similarity: {m['similarity_score']} | Source: {m['source']} (p.{m['page']})")
            print(f"  {m['text'][:100]}...\n")
    else:
        print(f"Directory '{pdf_dir}' not found.")