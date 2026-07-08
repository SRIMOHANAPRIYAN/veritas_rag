"""Run fixed 512 chunking ingestion pipeline for baseline comparison.

Ingests all documents from data/reconstructed/ (canonical text frame)
using a fixed 512-token chunking strategy. char_start/char_end offsets
are native to the reconstructed document text.
"""

import os
import sys
import hashlib
from pathlib import Path

import hydra
from omegaconf import DictConfig
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.ingestion.indexer import Indexer
from src.ingestion.metadata_tagger import Chunk


class FixedChunker:
    """Chunks documents into fixed-size token windows."""

    def __init__(self, model_name: str, chunk_size: int = 512, device: str = "cpu"):
        """Initialize fixed chunker with a sentence transformer tokenizer.

        Args:
            model_name: Name of the sentence transformer model (for tokenizer).
            chunk_size: Number of tokens per chunk.
            device: Device for the model.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.chunk_size = chunk_size

    def chunk_text(self, doc_id: str, full_text: str, doc_path: str) -> list:
        """Chunk a full document text into fixed-size token windows.

        char_start/char_end are character offsets into full_text.

        Args:
            doc_id: Document identifier.
            full_text: The full document text.
            doc_path: Path to the source file.

        Returns:
            List of Chunk objects with proper character offsets.
        """
        if not full_text.strip():
            return []

        chunks = []
        chunk_index = 0

        # Tokenize the full text to get token-to-char mapping
        encoding = self.model.tokenizer(
            full_text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        for i in range(0, len(token_ids), self.chunk_size):
            chunk_token_ids = token_ids[i : i + self.chunk_size]
            chunk_offsets = offsets[i : i + self.chunk_size]

            if not chunk_offsets:
                continue

            # Character offsets from the offset_mapping
            char_start = chunk_offsets[0][0]
            char_end = chunk_offsets[-1][1]

            text = full_text[char_start:char_end]
            if not text.strip():
                continue

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_fixed_{chunk_index:04d}",
                    text=text,
                    doc_id=doc_id,
                    doc_path=doc_path,
                    page=1,
                    heading_path=[],
                    is_table=False,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=len(chunk_token_ids),
                )
            )
            chunk_index += 1
        return chunks


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run fixed-512 ingestion from reconstructed corpus."""
    import torch

    logger.info(f"Interpreter: {sys.prefix}")
    logger.info(f"Torch version: {torch.__version__}")

    chunker_config_hash = hashlib.sha256(b"fixed_512").hexdigest()
    device = (
        "mps" if (hasattr(os, "uname") and os.uname().machine == "arm64") else "cpu"
    )

    logger.info("Initializing fixed chunker components...")
    chunker = FixedChunker(model_name=cfg.chunker.model, chunk_size=512, device=device)

    out_dir = Path("data/indexes_fixed512")
    # Always start fresh for a clean baseline
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
        logger.info(f"Cleared existing index directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"

    indexer = Indexer(
        faiss_path=str(out_dir / "faiss.index"),
        bm25_path=str(out_dir / "bm25.pkl"),
        metadata_db_path=str(out_dir / "metadata.db"),
        manifest_path=str(manifest_path),
        embedding_dim=cfg.indexer.embedding_dim,
        embedding_model=cfg.indexer.embedding_model,
        chunker_config_hash=chunker_config_hash,
    )

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        logger.error(
            f"Raw directory {raw_dir} does not exist. "
            "Please ensure corpus is downloaded."
        )
        return

    processed = 0
    skipped = 0
    failed = 0
    total_files = 0

    for file_path in sorted(raw_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in [".pdf", ".docx", ".html", ".txt"]:
            continue
            
        total_files += 1

        try:
            full_text = file_path.read_text(encoding="utf-8")
            doc_id = file_path.stem

            chunks = chunker.chunk_text(doc_id, full_text, str(file_path))
            if not chunks:
                logger.warning(f"No chunks produced for {file_path}")
                skipped += 1
                continue

            texts = [c.text for c in chunks]
            embeddings = chunker.model.encode(
                texts,
                batch_size=cfg.chunker.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            indexer.add_chunks(chunks, embeddings, file_path)
            processed += 1
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            failed += 1

    logger.info("=" * 60)
    logger.info("Fixed-512 Ingestion Summary:")
    logger.info(f"  Total files found:  {total_files}")
    logger.info(f"  Processed:          {processed}")
    logger.info(f"  Skipped (empty):    {skipped}")
    logger.info(f"  Failed:             {failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
