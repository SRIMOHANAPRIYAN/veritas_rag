"""Tests for semantic chunker."""

from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.document_parser import ParsedBlock


@pytest.fixture
def mock_model():
    with patch("src.ingestion.semantic_chunker.SentenceTransformer") as mock:
        model_instance = MagicMock()
        mock.return_value = model_instance

        # Mock tokenizer to return 1 token per word approx
        def mock_tokenize(text, **kwargs):
            words = text.split()
            # adding 2 for CLS and SEP to match logic
            return {"input_ids": [0] * (len(words) + 2)}

        model_instance.tokenizer = mock_tokenize

        # Mock encode to return deterministic vectors
        def mock_encode(sentences, **kwargs):
            # Return identity matrix like vectors or random
            # For testing boundaries, we want some to be similar and some not.
            # Let's say if sentences contain "apple", they are similar to each other.
            emb = []
            for s in sentences:
                if "apple" in s.lower():
                    emb.append([1.0, 0.0])
                else:
                    emb.append([0.0, 1.0])
            return np.array(emb, dtype=np.float32)

        model_instance.encode = mock_encode
        yield mock


def test_table_atomicity(mock_model):
    chunker = SemanticChunker("dummy", 0.5, 5, 20, 32)

    blocks = [
        ParsedBlock(
            text="Header1 | Header2\nVal1 | Val2",
            page=1,
            heading_path=[],
            is_table=True,
            source_path="test.pdf",
            char_start=0,
            char_end=31,
        )
    ]

    chunks = chunker.chunk_document("doc1", blocks)
    assert len(chunks) == 1
    assert chunks[0].is_table is True
    assert chunks[0].text == "Header1 | Header2\nVal1 | Val2"


def test_min_max_token_invariants(mock_model):
    chunker = SemanticChunker("dummy", 0.5, 5, 10, 32)

    blocks = [
        ParsedBlock(
            text="This is a short sentence. "
            * 15,  # 15 sentences, each 5 words -> 5 tokens
            page=1,
            heading_path=[],
            is_table=False,
            source_path="test.txt",
            char_start=0,
            char_end=390,
        )
    ]

    chunks = chunker.chunk_document("doc1", blocks)

    for chunk in chunks:
        assert chunk.token_count <= 10
        assert chunk.token_count >= 5


def test_cosine_boundary_logic(mock_model):
    chunker = SemanticChunker("dummy", 0.5, 2, 20, 32)

    blocks = [
        ParsedBlock(
            text="I like apple. Apple is good. I like banana. Banana is yellow.",
            page=1,
            heading_path=[],
            is_table=False,
            source_path="test.txt",
            char_start=0,
            char_end=61,
        )
    ]

    chunks = chunker.chunk_document("doc1", blocks)

    assert len(chunks) == 2
    assert chunks[0].text == "I like apple. Apple is good."
    assert chunks[1].text == "I like banana. Banana is yellow."


@pytest.mark.local_model
def test_continuous_stream_min_tokens():
    """Test that streaming sentences across blocks respects min_tokens."""
    chunker = SemanticChunker("all-mpnet-base-v2", 0.5, 10, 50, 32, device="cpu")
    # Provide dummy blocks
    blocks = []
    # Create 50 blocks of 5 tokens each
    for i in range(50):
        blocks.append(
            ParsedBlock(
                text="This is a test sentence.",
                page=1,
                heading_path=[],
                is_table=False,
                source_path="test.txt",
                char_start=i * 25,
                char_end=i * 25 + 24,
            )
        )
    
    chunks = chunker.chunk_document("doc1", blocks)
    
    # Check that less than 5% of chunks are under min_tokens (10)
    # The last chunk might be under min_tokens if it couldn't merge without exceeding max_tokens,
    # but the logic tries to merge.
    under_min = [c for c in chunks if c.token_count < 10]
    
    assert len(under_min) / max(1, len(chunks)) < 0.05


@pytest.mark.local_model
def test_real_model_initialization():
    """Test that the real model can load without crashing."""
    chunker = SemanticChunker("all-mpnet-base-v2", 0.5, 5, 20, 32, device="cpu")
    assert chunker.model is not None
