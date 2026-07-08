"""Tests for CrossEncoderReranker."""

import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.reranker import CrossEncoderReranker
from src.ingestion.metadata_tagger import Chunk


@pytest.fixture
def mock_chunk():
    return Chunk(
        chunk_id="test_0001",
        text="This is a test chunk.",
        doc_id="test_doc",
        doc_path="test.txt",
        page=1,
        heading_path=["Heading"],
        is_table=False,
        chunk_index=0,
        char_start=0,
        char_end=20,
        token_count=5,
    )


@pytest.fixture
def candidate(mock_chunk):
    return {"chunk": mock_chunk, "rrf_score": 0.5, "dense_rank": 1, "sparse_rank": 1}


@patch("src.retrieval.reranker.CrossEncoder")
def test_reranker_model_not_found(mock_ce, candidate):
    mock_ce.side_effect = Exception("Model not found")
    reranker = CrossEncoderReranker("dummy/path", top_k=2)

    assert reranker.model is None

    results = reranker.rerank("query", [candidate])
    assert len(results) == 1
    assert "rerank_score" not in results[0]


@patch("src.retrieval.reranker.CrossEncoder")
def test_reranker_prediction(mock_ce, candidate):
    mock_model = MagicMock()
    mock_model.predict.return_value = [2.0, -2.0]
    mock_ce.return_value = mock_model

    reranker = CrossEncoderReranker("dummy/path", top_k=2)

    c1 = {"chunk": candidate["chunk"], "id": 1}
    c2 = {"chunk": candidate["chunk"], "id": 2}

    results = reranker.rerank("query", [c1, c2])
    assert len(results) == 2
    assert results[0]["rerank_score"] > results[1]["rerank_score"]


@patch("src.retrieval.reranker.CrossEncoder")
def test_from_config_zeroshot(mock_ce):
    """from_config with mode='zeroshot' loads the zeroshot model."""
    mock_ce.return_value = MagicMock()
    reranker = CrossEncoderReranker.from_config(
        mode="zeroshot",
        zeroshot_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        finetuned_model="models/reranker/",
        top_k=8,
    )
    mock_ce.assert_called_once_with(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", device=reranker.device
    )
    assert reranker.model is not None


@patch("src.retrieval.reranker.CrossEncoder")
def test_from_config_finetuned(mock_ce):
    """from_config with mode='finetuned' loads the finetuned model."""
    mock_ce.return_value = MagicMock()
    reranker = CrossEncoderReranker.from_config(
        mode="finetuned",
        zeroshot_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        finetuned_model="models/reranker/",
        top_k=8,
    )
    mock_ce.assert_called_once_with("models/reranker/", device=reranker.device)
    assert reranker.model is not None


@pytest.mark.local_model
def test_reranker_real_model():
    """Test loading a small real model to verify pipeline."""
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert reranker.model is not None
