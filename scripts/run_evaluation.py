"""Run Phase 1 evaluation: compare semantic vs fixed chunking.

Supports both ID-matching and span-overlap evaluation modes.
Mode is controlled by config.yaml evaluation.span_mode.
"""

import os
import sys
import sqlite3
from pathlib import Path

import hydra
from omegaconf import DictConfig
from loguru import logger

from src.retrieval.hybrid_retriever import HybridRetriever
from evaluation.metrics import compare_chunking_strategies


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the Phase 1 evaluation comparison."""
    import torch

    logger.info(f"Interpreter: {sys.prefix}")
    logger.info(f"Torch version: {torch.__version__}")

    device = "mps" if (hasattr(os, "uname") and os.uname().machine == "arm64") else "cpu"

    logger.info("Loading semantic retriever...")
    semantic_retriever = HybridRetriever(
        faiss_path=cfg.indexer.faiss_index_path,
        bm25_path=cfg.indexer.bm25_index_path,
        metadata_db_path=cfg.indexer.metadata_db_path,
        manifest_path=str(Path(cfg.indexer.faiss_index_path).parent / "manifest.json"),
        model_name=cfg.indexer.embedding_model,
        rrf_k=cfg.retriever.rrf_k,
        top_k=cfg.retriever.top_k_fused,
        device=device,
    )

    logger.info("Loading fixed-512 retriever...")
    fixed_dir = Path("data/indexes_fixed512")
    fixed_retriever = HybridRetriever(
        faiss_path=str(fixed_dir / "faiss.index"),
        bm25_path=str(fixed_dir / "bm25.pkl"),
        metadata_db_path=str(fixed_dir / "metadata.db"),
        manifest_path=str(fixed_dir / "manifest.json"),
        model_name=cfg.indexer.embedding_model,
        rrf_k=cfg.retriever.rrf_k,
        top_k=cfg.retriever.top_k_fused,
        device=device,
    )

    golden_path = cfg.monitoring.golden_set_path
    output_path = "evaluation/benchmarks/results_phase1.json"

    # Log corpus statistics
    semantic_db = cfg.indexer.metadata_db_path
    fixed_db = str(fixed_dir / "metadata.db")

    with sqlite3.connect(semantic_db) as conn:
        sem_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sem_docs = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]
    with sqlite3.connect(fixed_db) as conn:
        fix_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fix_docs = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]

    logger.info(f"Semantic index: {sem_chunks} chunks, {sem_docs} distinct docs")
    logger.info(f"Fixed-512 index: {fix_chunks} chunks, {fix_docs} distinct docs")

    mode = "span" if cfg.evaluation.get("span_mode", False) else "id"
    logger.info(f"Evaluation mode: {mode}")

    if mode == "span":
        overlap_threshold = cfg.evaluation.get("span_overlap_threshold", 0.5)
        logger.info(f"Span overlap threshold: {overlap_threshold}")

        logger.info("Running span-overlap evaluation... (this may take a while)")
        results = compare_chunking_strategies(
            semantic_retriever,
            fixed_retriever,
            golden_path,
            output_path,
            mode="span",
            semantic_db_path=semantic_db,
            semantic_eval_db_path=semantic_db,
            fixed_eval_db_path=fixed_db,
            overlap_threshold=overlap_threshold,
        )
    else:
        logger.info("Running ID-match evaluation... (this may take a while)")
        results = compare_chunking_strategies(
            semantic_retriever,
            fixed_retriever,
            golden_path,
            output_path,
            mode="id",
        )

    logger.info("Evaluation complete.")
    logger.info(f"Semantic Chunking: {results['semantic_chunking']}")
    logger.info(f"Fixed-512 Chunking: {results['fixed_512_chunking']}")


if __name__ == "__main__":
    main()
