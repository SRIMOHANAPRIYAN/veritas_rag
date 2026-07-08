"""Semantic chunker for splitting documents into meaningful blocks."""

from typing import List
from dataclasses import dataclass
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.ingestion.document_parser import ParsedBlock
from src.ingestion.metadata_tagger import Chunk, MetadataTagger


def compute_cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between consecutive embeddings."""
    if len(embeddings) < 2:
        return np.array([])
    # Assuming embeddings are already L2 normalized
    return np.sum(embeddings[:-1] * embeddings[1:], axis=1)


@dataclass
class SentenceMeta:
    """Metadata for a single sentence in the document stream."""
    text: str
    char_start: int
    char_end: int
    page: int
    heading_path: List[str]
    source_path: str
    tokens: int


class SemanticChunker:
    """Chunks parsed blocks into semantic chunks using sentence embeddings."""

    def __init__(
        self,
        model_name: str,
        similarity_threshold: float,
        min_tokens: int,
        max_tokens: int,
        batch_size: int,
        device: str = "cpu",
    ):
        self.similarity_threshold = similarity_threshold
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.batch_size = batch_size

        logger.info("Loading spaCy model en_core_web_sm...")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2000000  # handle large blocks

        logger.info(f"Loading sentence transformer: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.tagger = MetadataTagger()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using the model's tokenizer."""
        tokens = self.model.tokenizer(text, truncation=False)["input_ids"]
        # Subtract 2 for CLS and SEP tokens
        return max(1, len(tokens) - 2)

    def _process_stream(self, doc_id: str, stream: List[SentenceMeta], start_idx: int) -> List[Chunk]:
        """Process a continuous stream of sentences into semantic chunks."""
        if not stream:
            return []

        # Batch encode all sentences in the stream
        embeddings = self.model.encode(
            [s.text for s in stream],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = compute_cosine_similarities(embeddings)

        chunks = []
        chunk_idx = start_idx

        current_chunk_sents = []
        current_chunk_tokens = 0

        for i, s_meta in enumerate(stream):
            # Check for hard cut at max_tokens
            if current_chunk_tokens + s_meta.tokens > self.max_tokens and current_chunk_sents:
                chunk_text = " ".join([s.text for s in current_chunk_sents])
                first_s = current_chunk_sents[0]
                last_s = current_chunk_sents[-1]
                
                chunks.append(
                    self.tagger.tag_chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        doc_path=first_s.source_path,
                        page=first_s.page,
                        heading_path=first_s.heading_path,
                        is_table=False,
                        chunk_index=chunk_idx,
                        char_start=first_s.char_start,
                        char_end=last_s.char_end,
                        token_count=current_chunk_tokens,
                    )
                )
                chunk_idx += 1
                current_chunk_sents = [s_meta]
                current_chunk_tokens = s_meta.tokens
                continue

            current_chunk_sents.append(s_meta)
            current_chunk_tokens += s_meta.tokens

            # Check if we should split AFTER this sentence
            if i < len(similarities):
                sim = similarities[i]
                # Boundary fires iff (cosine_sim < threshold AND current_tokens >= min_tokens)
                if sim < self.similarity_threshold and current_chunk_tokens >= self.min_tokens:
                    chunk_text = " ".join([s.text for s in current_chunk_sents])
                    first_s = current_chunk_sents[0]
                    last_s = current_chunk_sents[-1]
                    
                    chunks.append(
                        self.tagger.tag_chunk(
                            text=chunk_text,
                            doc_id=doc_id,
                            doc_path=first_s.source_path,
                            page=first_s.page,
                            heading_path=first_s.heading_path,
                            is_table=False,
                            chunk_index=chunk_idx,
                            char_start=first_s.char_start,
                            char_end=last_s.char_end,
                            token_count=current_chunk_tokens,
                        )
                    )
                    chunk_idx += 1
                    current_chunk_sents = []
                    current_chunk_tokens = 0

        # Flush remaining sentences
        if current_chunk_sents:
            chunk_text = " ".join([s.text for s in current_chunk_sents])
            first_s = current_chunk_sents[0]
            last_s = current_chunk_sents[-1]
            
            chunks.append(
                self.tagger.tag_chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    doc_path=first_s.source_path,
                    page=first_s.page,
                    heading_path=first_s.heading_path,
                    is_table=False,
                    chunk_index=chunk_idx,
                    char_start=first_s.char_start,
                    char_end=last_s.char_end,
                    token_count=current_chunk_tokens,
                )
            )
            chunk_idx += 1

        # Post-processing: a document's final chunk under min_tokens merges into the previous chunk
        if len(chunks) >= 2:
            last_chunk = chunks[-1]
            if last_chunk.token_count < self.min_tokens:
                prev_chunk = chunks[-2]
                new_tokens = prev_chunk.token_count + last_chunk.token_count

                if new_tokens <= self.max_tokens:
                    # Merge them
                    new_text = prev_chunk.text + " " + last_chunk.text
                    merged_chunk = self.tagger.tag_chunk(
                        text=new_text,
                        doc_id=doc_id,
                        doc_path=prev_chunk.doc_path,
                        page=prev_chunk.page,
                        heading_path=prev_chunk.heading_path,
                        is_table=prev_chunk.is_table,
                        chunk_index=prev_chunk.chunk_index,
                        char_start=prev_chunk.char_start,
                        char_end=last_chunk.char_end,
                        token_count=new_tokens,
                    )
                    chunks[-2] = merged_chunk
                    chunks.pop()

        return chunks

    def chunk_document(self, doc_id: str, blocks: List[ParsedBlock]) -> List[Chunk]:
        """Chunk a list of parsed blocks for a single document."""
        all_chunks = []
        chunk_idx = 0
        
        current_stream = []

        for block in blocks:
            if block.is_table:
                # Tables act as hard boundaries for the stream
                if current_stream:
                    stream_chunks = self._process_stream(doc_id, current_stream, chunk_idx)
                    all_chunks.extend(stream_chunks)
                    chunk_idx += len(stream_chunks)
                    current_stream = []

                chunk_text = block.text
                token_count = self._estimate_tokens(chunk_text)

                all_chunks.append(
                    self.tagger.tag_chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        doc_path=block.source_path,
                        page=block.page,
                        heading_path=block.heading_path,
                        is_table=True,
                        chunk_index=chunk_idx,
                        char_start=block.char_start,
                        char_end=block.char_end,
                        token_count=token_count,
                    )
                )
                chunk_idx += 1
                continue

            # Parse text into sentences and add to stream
            doc = self.nlp(block.text)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                
                s_meta = SentenceMeta(
                    text=sent_text,
                    char_start=block.char_start + sent.start_char,
                    char_end=block.char_start + sent.end_char,
                    page=block.page,
                    heading_path=block.heading_path,
                    source_path=block.source_path,
                    tokens=self._estimate_tokens(sent_text)
                )
                current_stream.append(s_meta)

        # Flush any remaining sentences in the stream
        if current_stream:
            stream_chunks = self._process_stream(doc_id, current_stream, chunk_idx)
            all_chunks.extend(stream_chunks)
            chunk_idx += len(stream_chunks)

        return all_chunks
