"""Hybrid retriever using FAISS (dense) and BM25 (sparse) with Reciprocal Rank Fusion."""
import json
import sqlite3
import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.common.exceptions import RetrievalError
from src.ingestion.metadata_tagger import Chunk


class HybridRetriever:
    """Retrieves top chunks using FAISS and BM25 combined via RRF."""

    def __init__(
        self,
        faiss_path: str,
        bm25_path: str,
        metadata_db_path: str,
        manifest_path: str,
        model_name: str,
        rrf_k: int = 60,
        top_k: int = 50,
        device: str = "cpu",
    ):
        self.faiss_path = Path(faiss_path)
        self.bm25_path = Path(bm25_path)
        self.metadata_db_path = Path(metadata_db_path)
        self.manifest_path = Path(manifest_path)
        self.chunk_ids_path = self.manifest_path.parent / "chunk_ids.json"

        self.rrf_k = rrf_k
        self.top_k = top_k
        self.model_name = model_name
        self.device = device

        self._load_and_validate_manifest()

        logger.info(f"Loading retriever with model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)

        self._load_indexes()

    def _load_and_validate_manifest(self):
        if not self.manifest_path.exists():
            raise RetrievalError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, "r") as f:
            self.manifest = json.load(f)

        if self.manifest.get("embedding_model") != self.model_name:
            raise RetrievalError(
                f"Model mismatch: config has {self.model_name}, "
                f"but manifest indicates {self.manifest.get('embedding_model')} was used."
            )

    def _load_indexes(self):
        if not self.faiss_path.exists() or not self.bm25_path.exists():
            raise RetrievalError("Indexes not found. Run ingestion first.")

        self.faiss_index = faiss.read_index(str(self.faiss_path))

        with open(self.bm25_path, "rb") as f:
            self.bm25_index = pickle.load(f)

        with open(self.chunk_ids_path, "r") as f:
            self.chunk_ids = json.load(f)

    def _fetch_chunks(self, chunk_ids: List[str]) -> Dict[str, Chunk]:
        """Fetch chunk metadata from SQLite."""
        if not chunk_ids:
            return {}

        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(chunk_ids))
            cursor.execute(
                f"SELECT chunk_id, text, doc_id, doc_path, page, heading_path, is_table, chunk_index, char_start, char_end, token_count "
                f"FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )

            results = {}
            for row in cursor.fetchall():
                chunk = Chunk(
                    chunk_id=row[0],
                    text=row[1],
                    doc_id=row[2],
                    doc_path=row[3],
                    page=row[4],
                    heading_path=json.loads(row[5]),
                    is_table=bool(row[6]),
                    chunk_index=row[7],
                    char_start=row[8],
                    char_end=row[9],
                    token_count=row[10],
                )
                results[chunk.chunk_id] = chunk

        return results

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top chunks using FAISS and BM25 combined via RRF."""
        if self.faiss_index.ntotal == 0:
            return []

        # Ensure we fetch enough candidates, but up to 100
        k_candidates = min(100, self.faiss_index.ntotal)

        # FAISS Retrieval (Dense)
        query_emb = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        _, faiss_indices = self.faiss_index.search(query_emb, k_candidates)

        faiss_hits = []
        for row_id in faiss_indices[0]:
            if row_id != -1 and row_id < len(self.chunk_ids):
                faiss_hits.append(self.chunk_ids[row_id])

        # BM25 Retrieval (Sparse)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Get top 100 indices from BM25
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k_candidates]
        bm25_hits = []
        for row_id in top_bm25_indices:
            if bm25_scores[row_id] > 0:  # Only consider non-zero scores
                bm25_hits.append(self.chunk_ids[row_id])

        # RRF Fusion
        rrf_scores = {}

        for rank_idx, chunk_id in enumerate(faiss_hits):
            rank = rank_idx + 1
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank_idx, chunk_id in enumerate(bm25_hits):
            rank = rank_idx + 1
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # Sort by RRF score descending
        sorted_hits = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_hits = sorted_hits[: self.top_k]

        if not top_hits:
            return []

        # Fetch metadata
        top_chunk_ids = [hit[0] for hit in top_hits]
        chunk_metadata = self._fetch_chunks(top_chunk_ids)

        # Build results
        results = []
        for chunk_id, score in top_hits:
            if chunk_id in chunk_metadata:
                results.append({"chunk": chunk_metadata[chunk_id], "score": score})

        return results
