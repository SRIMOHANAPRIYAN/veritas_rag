"""Run full ingestion pipeline."""

import os
from pathlib import Path
import hashlib
import json

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from src.ingestion.document_parser import DocumentParser
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.indexer import Indexer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg.chunker, resolve=True)
    config_str = json.dumps(config_dict, sort_keys=True)
    chunker_config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    device = (
        "mps" if (hasattr(os, "uname") and os.uname().machine == "arm64") else "cpu"
    )

    logger.info("Initializing components...")
    parser = DocumentParser()
    chunker = SemanticChunker(
        model_name=cfg.chunker.model,
        similarity_threshold=cfg.chunker.similarity_threshold,
        min_tokens=cfg.chunker.min_tokens,
        max_tokens=cfg.chunker.max_tokens,
        batch_size=cfg.chunker.batch_size,
        device=device,
    )

    manifest_path = Path(cfg.indexer.faiss_index_path).parent / "manifest.json"

    indexer = Indexer(
        faiss_path=cfg.indexer.faiss_index_path,
        bm25_path=cfg.indexer.bm25_index_path,
        metadata_db_path=cfg.indexer.metadata_db_path,
        manifest_path=str(manifest_path),
        embedding_dim=cfg.indexer.embedding_dim,
        embedding_model=cfg.indexer.embedding_model,
        chunker_config_hash=chunker_config_hash,
    )

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        logger.warning(f"Raw directory {raw_dir} does not exist.")
        return

    for file_path in raw_dir.rglob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() not in [".pdf", ".docx", ".html", ".txt"]:
                continue

            if not indexer.should_process_file(file_path):
                logger.info(f"Skipping unmodified file: {file_path}")
                continue

            logger.info(f"Processing {file_path}...")

            try:
                blocks = parser.parse(file_path)
                doc_id = file_path.stem

                chunks = chunker.chunk_document(doc_id, blocks)

                if not chunks:
                    logger.warning(f"No chunks produced for {file_path}")
                    continue

                texts = [c.text for c in chunks]
                embeddings = chunker.model.encode(
                    texts,
                    batch_size=cfg.chunker.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                indexer.add_chunks(chunks, embeddings, file_path)
                logger.info(f"Successfully indexed {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()
