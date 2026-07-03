"""Tests for metadata tagger."""

from src.ingestion.metadata_tagger import MetadataTagger, Chunk


def test_metadata_tagger():
    tagger = MetadataTagger()
    chunk = tagger.tag_chunk(
        text="Test text",
        doc_id="doc123",
        doc_path="/path/to/doc.pdf",
        page=1,
        heading_path=["Heading 1", "Heading 2"],
        is_table=False,
        chunk_index=42,
        char_start=100,
        char_end=109,
        token_count=2,
    )

    assert isinstance(chunk, Chunk)
    assert chunk.chunk_id == "doc123_0042"
    assert chunk.text == "Test text"
    assert chunk.doc_id == "doc123"
    assert chunk.doc_path == "/path/to/doc.pdf"
    assert chunk.page == 1
    assert chunk.heading_path == ["Heading 1", "Heading 2"]
    assert chunk.is_table is False
    assert chunk.chunk_index == 42
    assert chunk.char_start == 100
    assert chunk.char_end == 109
    assert chunk.token_count == 2
