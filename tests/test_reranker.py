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
        token_count=5
    )

@pytest.fixture
def candidate(mock_chunk):
    return {"chunk": mock_chunk, "rrf_score": 0.5, "dense_rank": 1, "sparse_rank": 1}

@patch("src.retrieval.reranker.CrossEncoder")
def test_reranker_model_not_found(mock_ce, candidate):
    mock_ce.side_effect = Exception("Model not found")
    reranker = CrossEncoderReranker("dummy/path", top_k=2)
    
    assert reranker.model is None
    
    # Should return original candidates unchanged
    results = reranker.rerank("query", [candidate])
    assert len(results) == 1
    assert "rerank_score" not in results[0]  # Shouldn't be added since model is None

@patch("src.retrieval.reranker.CrossEncoder")
def test_reranker_prediction(mock_ce, candidate):
    mock_model = MagicMock()
    # Logit scores: first item 2.0 (high), second item -2.0 (low)
    mock_model.predict.return_value = [2.0, -2.0]
    mock_ce.return_value = mock_model
    
    reranker = CrossEncoderReranker("dummy/path", top_k=2)
    
    # Create two candidates
    c1 = {"chunk": candidate["chunk"], "id": 1}
    c2 = {"chunk": candidate["chunk"], "id": 2}
    
    results = reranker.rerank("query", [c1, c2])
    
    assert len(results) == 2
    # c1 got 2.0 (sigmoid ~0.88), c2 got -2.0 (sigmoid ~0.11)
    # c1 should be first
    assert results[0]["id"] == 1
    assert "rerank_score" in results[0]
    assert results[0]["rerank_score"] > 0.5
    assert results[1]["rerank_score"] < 0.5

@pytest.mark.local_model
def test_reranker_real_model():
    """Test loading the real base model (downloads ~80MB)."""
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2)
    assert reranker.model is not None
