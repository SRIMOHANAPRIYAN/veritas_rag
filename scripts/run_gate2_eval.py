"""Gate 2 Evaluation: Reranker comparison (3 configs) + Classifier probe.

Evaluates hybrid retrieval with/without reranking on the golden set
using span-overlap mode.  Writes results_phase2.json.

MUST be run from project root with the venv activated:
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python scripts/run_gate2_eval.py
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

# ── interpreter guard ──
if "anaconda" in sys.prefix.lower():
    logger.error(f"Running under Anaconda ({sys.prefix}). Use the project venv.")
    sys.exit(1)

logger.info(f"Interpreter: {sys.prefix}")

# ── load config ──
cfg = OmegaConf.load("configs/config.yaml")

DEVICE = "mps" if (hasattr(os, "uname") and os.uname().machine == "arm64") else "cpu"
logger.info(f"Device: {DEVICE}")

# ── imports (after interpreter guard) ──
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from evaluation.metrics import evaluate_retriever_span

# ── paths ──
GOLDEN_SET_PATH = cfg.monitoring.golden_set_path
SEMANTIC_DB = cfg.indexer.metadata_db_path
OUTPUT_PATH = "evaluation/benchmarks/results_phase2.json"


# ── helpers ──
def count_index_stats(db_path: str) -> dict:
    """Return chunk count and distinct doc count."""
    with sqlite3.connect(db_path) as conn:
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        docs = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]
    return {"chunk_count": chunks, "doc_count": docs}


def run_retriever_eval(
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker | None,
    label: str,
) -> dict:
    """Evaluate a single config: retriever ± reranker, span mode."""
    logger.info(f"━━━ Evaluating config: {label} ━━━")

    # Monkey-patch retriever.retrieve to optionally rerank
    original_retrieve = retriever.retrieve

    if reranker is not None:
        def patched_retrieve(query: str):
            raw = original_retrieve(query)
            reranked = reranker.rerank(query, raw)
            return reranked
        retriever.retrieve = patched_retrieve

    t0 = time.time()
    result = evaluate_retriever_span(
        retriever,
        GOLDEN_SET_PATH,
        semantic_db_path=SEMANTIC_DB,
        eval_db_path=SEMANTIC_DB,
        overlap_threshold=cfg.evaluation.span_overlap_threshold,
        k_values=[5],
    )
    wall = time.time() - t0

    # Restore
    retriever.retrieve = original_retrieve

    metrics = result["metrics"]
    logger.info(
        f"  MRR={metrics['mrr']:.4f}  P@5={metrics['precision@5']:.4f}  "
        f"R@5={metrics['recall@5']:.4f}  wall={wall:.2f}s"
    )
    return {
        "label": label,
        "metrics": metrics,
        "span_collision_count": result["span_collision_count"],
        "skipped_queries": result["skipped_queries"],
        "effective_queries": result["effective_queries"],
        "wall_time_seconds": round(wall, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── Index stats ──
    idx_stats = count_index_stats(SEMANTIC_DB)
    logger.info(
        f"Semantic index: {idx_stats['chunk_count']} chunks, "
        f"{idx_stats['doc_count']} docs"
    )

    # ── Build retriever (top-50 for all configs) ──
    logger.info("Loading HybridRetriever (top_k=50)...")
    retriever = HybridRetriever(
        faiss_path=cfg.indexer.faiss_index_path,
        bm25_path=cfg.indexer.bm25_index_path,
        metadata_db_path=cfg.indexer.metadata_db_path,
        manifest_path=str(Path(cfg.indexer.faiss_index_path).parent / "manifest.json"),
        model_name=cfg.indexer.embedding_model,
        rrf_k=cfg.retriever.rrf_k,
        top_k=50,  # always feed 50 to reranker
        device=DEVICE,
    )

    # ── Config 1: Hybrid only (no reranker) ──
    config1 = run_retriever_eval(retriever, reranker=None, label="hybrid_only")

    # ── Config 2: Zero-shot cross-encoder ──
    logger.info("Loading zero-shot CrossEncoder (ms-marco-MiniLM-L-6-v2)...")
    zs_reranker = CrossEncoderReranker(
        model_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=8,
        device=DEVICE,
    )
    config2 = run_retriever_eval(retriever, reranker=zs_reranker, label="zero_shot_reranker")

    # ── Config 3: Fine-tuned cross-encoder ──
    logger.info("Loading fine-tuned CrossEncoder (models/reranker/)...")
    ft_reranker = CrossEncoderReranker(
        model_path="models/reranker/",
        top_k=8,
        device=DEVICE,
    )
    config3 = run_retriever_eval(retriever, reranker=ft_reranker, label="finetuned_reranker")

    # ── Gate 2 criterion ──
    gate2_delta = config3["metrics"]["mrr"] - config2["metrics"]["mrr"]
    gate2_pass = gate2_delta >= 0.08
    logger.info(f"Gate 2 Δ MRR (finetuned − zero_shot) = {gate2_delta:+.4f}  "
                f"{'PASS' if gate2_pass else 'FAIL'} (need ≥ +0.08)")

    # ── Assemble report ──
    report = {
        "gate": "GATE_2",
        "evaluation_mode": "span",
        "span_overlap_threshold": cfg.evaluation.span_overlap_threshold,
        "index_stats": idx_stats,
        "configs": [config1, config2, config3],
        "gate2_criterion": {
            "metric": "mrr",
            "finetuned_minus_zeroshot": round(gate2_delta, 4),
            "threshold": 0.08,
            "pass": gate2_pass,
        },
        "timestamp": time.time(),
    }

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
