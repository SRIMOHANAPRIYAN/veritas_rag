"""Semantic chunker for splitting documents into meaningful blocks."""

from typing import List
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

    def chunk_document(self, doc_id: str, blocks: List[ParsedBlock]) -> List[Chunk]:
        """Chunk a list of parsed blocks for a single document."""
        all_chunks = []
        chunk_idx = 0

        for block in blocks:
            if block.is_table:
                # Tables bypass splitting entirely
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
                        char_start=0,
                        char_end=len(chunk_text),
                        token_count=token_count,
                    )
                )
                chunk_idx += 1
                continue

            # For text, we do semantic chunking
            doc = self.nlp(block.text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not sentences:
                continue

            if len(sentences) == 1:
                # Only one sentence
                chunk_text = sentences[0]
                token_count = self._estimate_tokens(chunk_text)
                all_chunks.append(
                    self.tagger.tag_chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        doc_path=block.source_path,
                        page=block.page,
                        heading_path=block.heading_path,
                        is_table=False,
                        chunk_index=chunk_idx,
                        char_start=0,
                        char_end=len(chunk_text),
                        token_count=token_count,
                    )
                )
                chunk_idx += 1
                continue

            # Batch encode
            embeddings = self.model.encode(
                sentences,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            similarities = compute_cosine_similarities(embeddings)

            current_chunk_sentences = []
            current_chunk_tokens = 0

            char_offset = 0

            for i, sentence in enumerate(sentences):
                sent_tokens = self._estimate_tokens(sentence)

                # Check for hard cut at max_tokens
                if (
                    current_chunk_tokens + sent_tokens > self.max_tokens
                    and current_chunk_sentences
                ):
                    # Force a boundary before this sentence
                    chunk_text = " ".join(current_chunk_sentences)
                    all_chunks.append(
                        self.tagger.tag_chunk(
                            text=chunk_text,
                            doc_id=doc_id,
                            doc_path=block.source_path,
                            page=block.page,
                            heading_path=block.heading_path,
                            is_table=False,
                            chunk_index=chunk_idx,
                            char_start=char_offset,
                            char_end=char_offset + len(chunk_text),
                            token_count=current_chunk_tokens,
                        )
                    )
                    chunk_idx += 1
                    char_offset += len(chunk_text) + 1  # +1 for space

                    current_chunk_sentences = [sentence]
                    current_chunk_tokens = sent_tokens
                    continue

                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sent_tokens

                # Check if we should split AFTER this sentence
                if i < len(similarities):
                    sim = similarities[i]
                    # Boundary fires iff (cosine_sim < threshold AND current_tokens >= min_tokens)
                    if (
                        sim < self.similarity_threshold
                        and current_chunk_tokens >= self.min_tokens
                    ):
                        # Split!
                        chunk_text = " ".join(current_chunk_sentences)
                        all_chunks.append(
                            self.tagger.tag_chunk(
                                text=chunk_text,
                                doc_id=doc_id,
                                doc_path=block.source_path,
                                page=block.page,
                                heading_path=block.heading_path,
                                is_table=False,
                                chunk_index=chunk_idx,
                                char_start=char_offset,
                                char_end=char_offset + len(chunk_text),
                                token_count=current_chunk_tokens,
                            )
                        )
                        chunk_idx += 1
                        char_offset += len(chunk_text) + 1

                        current_chunk_sentences = []
                        current_chunk_tokens = 0

            # Flush remaining sentences
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                all_chunks.append(
                    self.tagger.tag_chunk(
                        text=chunk_text,
                        doc_id=doc_id,
                        doc_path=block.source_path,
                        page=block.page,
                        heading_path=block.heading_path,
                        is_table=False,
                        chunk_index=chunk_idx,
                        char_start=char_offset,
                        char_end=char_offset + len(chunk_text),
                        token_count=current_chunk_tokens,
                    )
                )
                chunk_idx += 1

        # Post-processing: a document's final chunk under min_tokens merges into the previous chunk.
        if len(all_chunks) >= 2:
            last_chunk = all_chunks[-1]
            if last_chunk.token_count < self.min_tokens:
                prev_chunk = all_chunks[-2]
                new_tokens = prev_chunk.token_count + last_chunk.token_count

                if new_tokens <= self.max_tokens:
                    # Merge them
                    new_text = prev_chunk.text + "\n\n" + last_chunk.text

                    merged_chunk = self.tagger.tag_chunk(
                        text=new_text,
                        doc_id=doc_id,
                        doc_path=prev_chunk.doc_path,
                        page=prev_chunk.page,
                        heading_path=prev_chunk.heading_path,
                        is_table=prev_chunk.is_table,
                        chunk_index=prev_chunk.chunk_index,
                        char_start=prev_chunk.char_start,
                        char_end=prev_chunk.char_start + len(new_text),
                        token_count=new_tokens,
                    )
                    all_chunks[-2] = merged_chunk
                    all_chunks.pop()

        return all_chunks
