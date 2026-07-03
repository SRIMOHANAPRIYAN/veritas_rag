"""Chunk metadata schema and tagger."""

from typing import List
from pydantic import BaseModel


class Chunk(BaseModel):
    """A semantic chunk of text with complete metadata for traceability."""

    chunk_id: str
    text: str
    doc_id: str
    doc_path: str
    page: int
    heading_path: List[str]
    is_table: bool
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int


class MetadataTagger:
    """Tags text chunks with metadata to create standardized Chunk objects."""

    def tag_chunk(
        self,
        text: str,
        doc_id: str,
        doc_path: str,
        page: int,
        heading_path: List[str],
        is_table: bool,
        chunk_index: int,
        char_start: int,
        char_end: int,
        token_count: int,
    ) -> Chunk:
        """Create a Chunk object with a formatted chunk_id."""
        chunk_id = f"{doc_id}_{chunk_index:04d}"
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            doc_id=doc_id,
            doc_path=doc_path,
            page=page,
            heading_path=heading_path,
            is_table=is_table,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
            token_count=token_count,
        )
