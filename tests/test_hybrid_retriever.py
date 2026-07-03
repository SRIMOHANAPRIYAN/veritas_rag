"""Tests for hybrid retriever."""
import json
import pickle
import sqlite3
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import faiss
from rank_bm25 import BM25Okapi

from src.retrieval.hybrid_retriever import HybridRetriever
from src.common.exceptions import RetrievalError


@pytest.fixture
def setup_indexes(tmp_path):
    faiss_path = tmp_path / "indexes" / "faiss.index"
    bm25_path = tmp_path / "indexes" / "bm25.pkl"
    metadata_db_path = tmp_path / "indexes" / "metadata.db"
    manifest_path = tmp_path / "indexes" / "manifest.json"
    chunk_ids_path = tmp_path / "indexes" / "chunk_ids.json"
    
    tmp_path.joinpath("indexes").mkdir()
    
    # Manifest
    with open(manifest_path, "w") as f:
        json.dump({
            "embedding_model": "dummy-model",
            "version": "1.0"
        }, f)
        
    # Chunk IDs
    chunk_ids = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    with open(chunk_ids_path, "w") as f:
        json.dump(chunk_ids, f)
        
    # FAISS
    index = faiss.IndexFlatIP(4)
    # Add dummy vectors
    index.add(np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0]
    ], dtype=np.float32))
    faiss.write_index(index, str(faiss_path))
    
    # BM25
    bm25 = BM25Okapi([["apple"], ["banana"], ["apple", "banana"], ["orange"], ["grape"]])
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
        
    # SQLite
    with sqlite3.connect(metadata_db_path) as conn:
        conn.execute(
            """
            CREATE TABLE chunks (
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
        cursor = conn.cursor()
        for i, cid in enumerate(chunk_ids):
            cursor.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, f"text {i}", "doc1", "doc.txt", 1, "[]", False, i, 0, 5, 2, i)
            )
            
    return tmp_path / "indexes"

@patch("src.retrieval.hybrid_retriever.SentenceTransformer")
def test_manifest_mismatch(mock_st, setup_indexes):
    with pytest.raises(RetrievalError, match="Model mismatch"):
        HybridRetriever(
            faiss_path=str(setup_indexes / "faiss.index"),
            bm25_path=str(setup_indexes / "bm25.pkl"),
            metadata_db_path=str(setup_indexes / "metadata.db"),
            manifest_path=str(setup_indexes / "manifest.json"),
            model_name="different-model"
        )

@patch("src.retrieval.hybrid_retriever.SentenceTransformer")
def test_rrf_math(mock_st, setup_indexes):
    # Setup mock encode
    mock_instance = MagicMock()
    # Let query vector match chunk1 best (row 0)
    mock_instance.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mock_st.return_value = mock_instance
    
    retriever = HybridRetriever(
        faiss_path=str(setup_indexes / "faiss.index"),
        bm25_path=str(setup_indexes / "bm25.pkl"),
        metadata_db_path=str(setup_indexes / "metadata.db"),
        manifest_path=str(setup_indexes / "manifest.json"),
        model_name="dummy-model",
        rrf_k=60,
        top_k=2
    )
    
    # Query: "banana"
    # FAISS will return chunk1 (rank 1), chunk2 (rank 2), chunk3 (rank 3)
    # BM25 for "banana" will return chunk2 (highest TF-IDF), chunk3 (second), chunk1 (zero)
    # chunk1: FAISS rank 1 -> 1/61. BM25 rank none. Score = 1/61 = 0.01639
    # chunk2: FAISS rank 2 -> 1/62. BM25 rank 1 -> 1/61. Score = 1/62 + 1/61 = 0.0161 + 0.01639 = 0.0325
    # chunk3: FAISS rank 3 -> 1/63. BM25 rank 2 -> 1/62. Score = 1/63 + 1/62 = 0.0158 + 0.0161 = 0.0320
    
    results = retriever.retrieve("banana")
    
    assert len(results) == 2
    assert results[0]["chunk"].chunk_id == "chunk2"
    assert results[1]["chunk"].chunk_id == "chunk3"
