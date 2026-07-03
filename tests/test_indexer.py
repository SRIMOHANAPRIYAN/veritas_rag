"""Tests for Indexer."""

import sqlite3
import numpy as np
import pytest

from src.ingestion.indexer import Indexer
from src.ingestion.metadata_tagger import Chunk


@pytest.fixture
def temp_indexer(tmp_path):
    faiss_path = tmp_path / "indexes" / "faiss.index"
    bm25_path = tmp_path / "indexes" / "bm25.pkl"
    metadata_db_path = tmp_path / "indexes" / "metadata.db"
    manifest_path = tmp_path / "indexes" / "manifest.json"

    return Indexer(
        faiss_path=str(faiss_path),
        bm25_path=str(bm25_path),
        metadata_db_path=str(metadata_db_path),
        manifest_path=str(manifest_path),
        embedding_dim=4,
        embedding_model="dummy-model",
        chunker_config_hash="abc",
    )


def test_indexer_initialization(temp_indexer):
    assert temp_indexer.faiss_index.ntotal == 0
    assert temp_indexer.bm25_index is None
    assert len(temp_indexer.chunk_ids) == 0
    assert temp_indexer.manifest["embedding_model"] == "dummy-model"


def test_idempotent_reingestion(temp_indexer, tmp_path):
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("Hello world.")

    # First time, should process
    assert temp_indexer.should_process_file(doc_file) is True

    chunks = [
        Chunk(
            chunk_id="doc_0000",
            text="Hello world.",
            doc_id="doc",
            doc_path=str(doc_file),
            page=1,
            heading_path=[],
            is_table=False,
            chunk_index=0,
            char_start=0,
            char_end=12,
            token_count=3,
        )
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    temp_indexer.add_chunks(chunks, embeddings, doc_file)

    # Second time, should NOT process because of hash match
    assert temp_indexer.should_process_file(doc_file) is False

    # If file changes, it should process
    doc_file.write_text("Hello world modified.")
    assert temp_indexer.should_process_file(doc_file) is True


def test_add_chunks(temp_indexer, tmp_path):
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("Hello world.")

    chunks = [
        Chunk(
            chunk_id="doc_0000",
            text="Hello world.",
            doc_id="doc",
            doc_path=str(doc_file),
            page=1,
            heading_path=[],
            is_table=False,
            chunk_index=0,
            char_start=0,
            char_end=12,
            token_count=3,
        )
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    temp_indexer.add_chunks(chunks, embeddings, doc_file)

    # FAISS has 1 element
    assert temp_indexer.faiss_index.ntotal == 1

    # BM25 is built
    assert temp_indexer.bm25_index is not None
    assert temp_indexer.bm25_index.corpus_size == 1

    # SQLite has 1 row
    with sqlite3.connect(temp_indexer.metadata_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, faiss_row_id FROM chunks")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "doc_0000"
        assert rows[0][1] == 0

    # Chunk IDs maintained
    assert temp_indexer.chunk_ids == ["doc_0000"]

    # Save was called
    assert temp_indexer.faiss_path.exists()
    assert temp_indexer.bm25_path.exists()
    assert temp_indexer.manifest_path.exists()
    assert temp_indexer.chunk_ids_path.exists()


def test_load_existing_indexes(temp_indexer, tmp_path):
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("Hello world.")

    chunks = [
        Chunk(
            chunk_id="doc_0000",
            text="Hello world.",
            doc_id="doc",
            doc_path=str(doc_file),
            page=1,
            heading_path=[],
            is_table=False,
            chunk_index=0,
            char_start=0,
            char_end=12,
            token_count=3,
        )
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    temp_indexer.add_chunks(chunks, embeddings, doc_file)

    # Now create a new Indexer pointing to the same paths
    new_indexer = Indexer(
        faiss_path=str(temp_indexer.faiss_path),
        bm25_path=str(temp_indexer.bm25_path),
        metadata_db_path=str(temp_indexer.metadata_db_path),
        manifest_path=str(temp_indexer.manifest_path),
        embedding_dim=4,
        embedding_model="dummy-model",
        chunker_config_hash="abc",
    )

    assert new_indexer.faiss_index.ntotal == 1
    assert new_indexer.chunk_ids == ["doc_0000"]
    assert new_indexer.bm25_index is not None
    assert new_indexer.manifest["embedding_model"] == "dummy-model"
