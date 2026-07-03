"""Indexer for vector, sparse, and metadata stores."""

import json
import sqlite3
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from loguru import logger

from src.ingestion.metadata_tagger import Chunk


class Indexer:
    """Manages FAISS, BM25, and SQLite metadata indices."""

    def __init__(
        self,
        faiss_path: str,
        bm25_path: str,
        metadata_db_path: str,
        manifest_path: str,
        embedding_dim: int,
        embedding_model: str,
        chunker_config_hash: str,
    ):
        self.faiss_path = Path(faiss_path)
        self.bm25_path = Path(bm25_path)
        self.metadata_db_path = Path(metadata_db_path)
        self.manifest_path = Path(manifest_path)
        self.chunk_ids_path = self.manifest_path.parent / "chunk_ids.json"

        # Ensure directories exist
        for p in [
            self.faiss_path,
            self.bm25_path,
            self.metadata_db_path,
            self.manifest_path,
            self.chunk_ids_path,
        ]:
            p.parent.mkdir(parents=True, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.chunker_config_hash = chunker_config_hash

        self.version = "1.0"

        self.faiss_index = None
        self.bm25_index = None
        self.chunk_ids = []
        self.manifest = self._load_manifest()

        self._init_sqlite()
        self._load_indexes()

    def _init_sqlite(self):
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT,
                    doc_id TEXT,
                    doc_path TEXT,
                    page INTEGER,
                    heading_path TEXT,
                    is_table BOOLEAN,
                    chunk_index INTEGER,
                    char_start INTEGER,
                    char_end INTEGER,
                    token_count INTEGER,
                    faiss_row_id INTEGER
                )
                """
            )

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {
            "version": self.version,
            "embedding_model": self.embedding_model,
            "chunker_config_hash": self.chunker_config_hash,
            "per_file_sha256": {},
            "created_at": datetime.utcnow().isoformat(),
        }

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _load_indexes(self):
        if self.faiss_path.exists():
            self.faiss_index = faiss.read_index(str(self.faiss_path))
        else:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        if self.bm25_path.exists():
            with open(self.bm25_path, "rb") as f:
                self.bm25_index = pickle.load(f)

        if self.chunk_ids_path.exists():
            with open(self.chunk_ids_path, "r") as f:
                self.chunk_ids = json.load(f)

    def _save_indexes(self):
        faiss.write_index(self.faiss_index, str(self.faiss_path))
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25_index, f)
        with open(self.chunk_ids_path, "w") as f:
            json.dump(self.chunk_ids, f)

    def compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed (idempotency check)."""
        if not file_path.exists():
            return False
        current_hash = self.compute_sha256(file_path)
        stored_hash = self.manifest["per_file_sha256"].get(str(file_path))
        return current_hash != stored_hash

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray, file_path: Path):
        """Add chunks for a file, update indexes, and save to disk."""
        if len(chunks) == 0:
            return

        assert len(chunks) == len(embeddings), "Chunks and embeddings length mismatch."

        # 1. Add to FAISS
        start_row_id = self.faiss_index.ntotal
        self.faiss_index.add(embeddings)

        # 2. Update chunk_ids list
        new_chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_ids.extend(new_chunk_ids)

        # 3. Add to SQLite
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                row_id = start_row_id + i
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, text, doc_id, doc_path, page, heading_path, is_table, chunk_index, char_start, char_end, token_count, faiss_row_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.text,
                        chunk.doc_id,
                        chunk.doc_path,
                        chunk.page,
                        json.dumps(chunk.heading_path),
                        chunk.is_table,
                        chunk.chunk_index,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.token_count,
                        row_id,
                    ),
                )

        # 4. Update BM25 (rebuild entirely)
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.cursor()
            # Must fetch in exact faiss_row_id order to align with self.chunk_ids
            cursor.execute("SELECT text FROM chunks ORDER BY faiss_row_id ASC")
            all_texts = [row[0] for row in cursor.fetchall()]

        tokenized_corpus = [text.lower().split() for text in all_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # 5. Update manifest
        self.manifest["per_file_sha256"][str(file_path)] = self.compute_sha256(
            file_path
        )

        # 6. Save to disk
        self._save_indexes()
        self._save_manifest()

        logger.info(f"Indexed {len(chunks)} chunks for {file_path}")
