"""Tests for evaluation metrics."""

import json
from unittest.mock import MagicMock

from evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    evaluate_retriever,
    create_golden_set_scaffold,
)
from src.ingestion.metadata_tagger import Chunk


def test_metrics_math():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc2", "doc4"]

    # Precision@2 -> retrieved[:2] = ["doc1", "doc2"], relevant=["doc2", "doc4"]
    # hits = "doc2". precision = 1/2 = 0.5
    assert calculate_precision_at_k(retrieved, relevant, 2) == 0.5

    # Recall@2 -> hits = 1. recall = 1/2 = 0.5
    assert calculate_recall_at_k(retrieved, relevant, 2) == 0.5

    # MRR -> first relevant is doc2 at index 1 (rank 2). 1/2 = 0.5
    assert calculate_mrr(retrieved, relevant) == 0.5


def test_evaluate_retriever(tmp_path):
    golden_path = tmp_path / "golden.json"
    create_golden_set_scaffold(str(golden_path))

    # Update scaffold to test our specific logic
    with open(golden_path, "r") as f:
        data = json.load(f)

    data["queries"] = [
        {"query": "q1", "relevant_doc_ids": ["doc2", "doc4"]},
        {"query": "q2", "relevant_doc_ids": ["doc1"]},
    ]
    with open(golden_path, "w") as f:
        json.dump(data, f)

    mock_retriever = MagicMock()

    def mock_retrieve(q):
        if q == "q1":
            return [
                {
                    "chunk": Chunk(
                        chunk_id="c1",
                        text="",
                        doc_id="doc1",
                        doc_path="",
                        page=1,
                        heading_path=[],
                        is_table=False,
                        chunk_index=1,
                        char_start=0,
                        char_end=0,
                        token_count=0,
                    ),
                    "score": 1.0,
                },
                {
                    "chunk": Chunk(
                        chunk_id="c2",
                        text="",
                        doc_id="doc2",
                        doc_path="",
                        page=1,
                        heading_path=[],
                        is_table=False,
                        chunk_index=1,
                        char_start=0,
                        char_end=0,
                        token_count=0,
                    ),
                    "score": 0.9,
                },
                {
                    "chunk": Chunk(
                        chunk_id="c3",
                        text="",
                        doc_id="doc3",
                        doc_path="",
                        page=1,
                        heading_path=[],
                        is_table=False,
                        chunk_index=1,
                        char_start=0,
                        char_end=0,
                        token_count=0,
                    ),
                    "score": 0.8,
                },
            ]
        else:
            return [
                {
                    "chunk": Chunk(
                        chunk_id="c4",
                        text="",
                        doc_id="doc3",
                        doc_path="",
                        page=1,
                        heading_path=[],
                        is_table=False,
                        chunk_index=1,
                        char_start=0,
                        char_end=0,
                        token_count=0,
                    ),
                    "score": 1.0,
                },
                {
                    "chunk": Chunk(
                        chunk_id="c5",
                        text="",
                        doc_id="doc4",
                        doc_path="",
                        page=1,
                        heading_path=[],
                        is_table=False,
                        chunk_index=1,
                        char_start=0,
                        char_end=0,
                        token_count=0,
                    ),
                    "score": 0.9,
                },
            ]

    mock_retriever.retrieve = mock_retrieve

    results = evaluate_retriever(mock_retriever, str(golden_path), k_values=[2])

    assert results["precision@2"] == 0.25
    assert results["recall@2"] == 0.25
    assert results["mrr"] == 0.25
