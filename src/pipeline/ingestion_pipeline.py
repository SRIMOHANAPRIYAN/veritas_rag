"""Unified orchestration for the ingestion pipeline."""

import json
import hashlib
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf
from loguru import logger

from src.ingestion.document_parser import DocumentParser
from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.indexer import Indexer
from src.common.exceptions import IngestionError
from src.common.device import DEVICE

class IngestionPipeline:
    """Orchestrates parsing, chunking, and indexing with per-stage error capture."""
    
    def __init__(self):
        self.cfg = OmegaConf.load("configs/config.yaml")
        
        config_dict = OmegaConf.to_container(self.cfg.chunker, resolve=True)
        config_str = json.dumps(config_dict, sort_keys=True)
        self.chunker_config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        self.device = DEVICE
        
        logger.info("Initializing IngestionPipeline components...")
        self.parser = DocumentParser()
        self.chunker = SemanticChunker(
            model_name=self.cfg.chunker.model,
            similarity_threshold=self.cfg.chunker.similarity_threshold,
            min_tokens=self.cfg.chunker.min_tokens,
            max_tokens=self.cfg.chunker.max_tokens,
            batch_size=self.cfg.chunker.batch_size,
            device=self.device,
        )
        
        manifest_path = Path(self.cfg.indexer.faiss_index_path).parent / "manifest.json"
        
        self.indexer = Indexer(
            faiss_path=self.cfg.indexer.faiss_index_path,
            bm25_path=self.cfg.indexer.bm25_index_path,
            metadata_db_path=self.cfg.indexer.metadata_db_path,
            manifest_path=str(manifest_path),
            embedding_dim=self.cfg.indexer.embedding_dim,
            embedding_model=self.cfg.indexer.embedding_model,
            chunker_config_hash=self.chunker_config_hash,
        )

    def process_directory(self, raw_dir: str = "data/raw") -> dict:
        """Process an entire directory of documents."""
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            raise IngestionError(f"Raw directory {raw_dir} does not exist.")
            
        stats = {"processed": 0, "skipped": 0, "failed": 0}
        
        for file_path in raw_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in [".pdf", ".docx", ".html", ".txt"]:
                continue
                
            if not self.indexer.should_process_file(file_path):
                logger.info(f"Skipping unmodified file: {file_path}")
                stats["skipped"] += 1
                continue
                
            try:
                self.process_file(file_path)
                stats["processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                stats["failed"] += 1
                
        logger.info(f"Ingestion complete: {stats}")
        return stats

    def process_file(self, file_path: Path):
        """Process a single file through the pipeline."""
        logger.info(f"Processing {file_path}...")
        
        # 1. Parse
        try:
            blocks = self.parser.parse(file_path)
        except Exception as e:
            raise IngestionError(f"Parsing failed for {file_path}: {e}") from e
            
        # 2. Chunk
        try:
            doc_id = file_path.stem
            chunks = self.chunker.chunk_document(doc_id, blocks)
        except Exception as e:
            raise IngestionError(f"Chunking failed for {file_path}: {e}") from e
            
        if not chunks:
            logger.warning(f"No chunks produced for {file_path}")
            return
            
        # 3. Index
        try:
            texts = [c.text for c in chunks]
            embeddings = self.chunker.model.encode(
                texts,
                batch_size=self.cfg.chunker.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.indexer.add_chunks(chunks, embeddings, file_path)
            logger.info(f"Successfully indexed {file_path} ({len(chunks)} chunks).")
        except Exception as e:
            raise IngestionError(f"Indexing failed for {file_path}: {e}") from e
