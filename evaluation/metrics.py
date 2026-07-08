"""Evaluation metrics for RAG retrieval.

Supports two evaluation modes:
- "id": Exact chunk ID matching (original mode).
- "span": Span-overlap matching using (doc_id, char_start, char_end) with a
  configurable overlap threshold. Both strategies are scored against the same
  canonical text frame (data/reconstructed/).
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def calculate_precision_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """Calculate Precision@K."""
    if not retrieved_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def calculate_recall_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """Calculate Recall@K."""
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_ids)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank for a single query."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Span-overlap functions
# ---------------------------------------------------------------------------


def compute_span_overlap(
    gold_start: int, gold_end: int, ret_start: int, ret_end: int
) -> float:
    """Compute overlap ratio: overlap_chars / gold_span_length.

    Args:
        gold_start: Start character offset of the gold span.
        gold_end: End character offset of the gold span.
        ret_start: Start character offset of the retrieved span.
        ret_end: End character offset of the retrieved span.

    Returns:
        Overlap ratio in [0.0, 1.0]. Returns 0.0 if gold span has zero length.
    """
    gold_length = gold_end - gold_start
    if gold_length <= 0:
        return 0.0
    overlap_start = max(gold_start, ret_start)
    overlap_end = min(gold_end, ret_end)
    overlap_chars = max(0, overlap_end - overlap_start)
    return overlap_chars / gold_length


def is_span_relevant(
    gold_doc_id: str,
    gold_start: int,
    gold_end: int,
    ret_doc_id: str,
    ret_start: int,
    ret_end: int,
    threshold: float,
) -> bool:
    """Check if a retrieved chunk is relevant via span overlap.

    A retrieved chunk is relevant iff:
    1. Same doc_id as the gold chunk, AND
    2. overlap_chars / gold_span_length >= threshold.

    Args:
        gold_doc_id: Document ID of the gold chunk.
        gold_start: Start char offset of the gold chunk.
        gold_end: End char offset of the gold chunk.
        ret_doc_id: Document ID of the retrieved chunk.
        ret_start: Start char offset of the retrieved chunk.
        ret_end: End char offset of the retrieved chunk.
        threshold: Minimum overlap ratio to consider relevant.

    Returns:
        True if the retrieved chunk is relevant.
    """
    if gold_doc_id != ret_doc_id:
        return False
    overlap = compute_span_overlap(gold_start, gold_end, ret_start, ret_end)
    return overlap >= threshold


def resolve_gold_span_from_db(
    chunk_id: str,
    semantic_db_path: str,
) -> Tuple[str, int, int]:
    """Resolve a gold chunk's span directly from the database.

    Args:
        chunk_id: The gold chunk ID from the golden set.
        semantic_db_path: Path to the semantic metadata.db.

    Returns:
        Tuple of (doc_id, char_start, char_end).

    Raises:
        ValueError: If the chunk is not found in the DB.
    """
    with sqlite3.connect(semantic_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT doc_id, char_start, char_end FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Chunk {chunk_id} not found in {semantic_db_path}")

    return row[0], row[1], row[2]


# ---------------------------------------------------------------------------
# ID-mode evaluator (original)
# ---------------------------------------------------------------------------


def evaluate_retriever(
    retriever: Any,
    golden_set_path: str,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate a retriever against a golden set using ID matching."""
    if k_values is None:
        k_values = [5]

    with open(golden_set_path, "r") as f:
        golden_set = json.load(f)

    queries = golden_set.get("queries", [])
    if not queries:
        return {}

    results: Dict[str, float] = {"mrr": 0.0}
    for k in k_values:
        results[f"precision@{k}"] = 0.0
        results[f"recall@{k}"] = 0.0

    for q in queries:
        query_text = q["query"]
        relevant_chunks = q["relevant_chunk_ids"]

        retrieved = retriever.retrieve(query_text)

        # Deduplicate by chunk_id
        retrieved_chunks: List[str] = []
        seen: set = set()
        for hit in retrieved:
            chunk_id = hit["chunk"].chunk_id
            if chunk_id not in seen:
                seen.add(chunk_id)
                retrieved_chunks.append(chunk_id)

        results["mrr"] += calculate_mrr(retrieved_chunks, relevant_chunks)
        for k in k_values:
            results[f"precision@{k}"] += calculate_precision_at_k(
                retrieved_chunks, relevant_chunks, k
            )
            results[f"recall@{k}"] += calculate_recall_at_k(
                retrieved_chunks, relevant_chunks, k
            )

    num_queries = len(queries)
    results["mrr"] /= num_queries
    for k in k_values:
        results[f"precision@{k}"] /= num_queries
        results[f"recall@{k}"] /= num_queries

    return results


# ---------------------------------------------------------------------------
# Span-mode evaluator
# ---------------------------------------------------------------------------


def evaluate_retriever_span(
    retriever: Any,
    golden_set_path: str,
    semantic_db_path: str,
    eval_db_path: str,
    overlap_threshold: float = 0.5,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Evaluate a retriever using span-overlap relevance.

    For each golden query:
    1. Resolve the gold chunk's span from the semantic db.
    2. For each retrieved chunk, look up its (doc_id, char_start, char_end).
    3. Score relevance via is_span_relevant.

    Args:
        retriever: A retriever with a .retrieve(query) method.
        golden_set_path: Path to golden_set.json.
        semantic_db_path: Path to semantic metadata.db (for gold spans).
        eval_db_path: Path to the retriever's metadata.db (for retrieved spans).
        overlap_threshold: Minimum overlap ratio for relevance.
        k_values: List of K values for precision/recall.

    Returns:
        Dict with metrics and span_collision_count.
    """
    if k_values is None:
        k_values = [5]

    with open(golden_set_path, "r") as f:
        golden_set = json.load(f)

    queries = golden_set.get("queries", [])
    if not queries:
        return {}

    metrics: Dict[str, float] = {"mrr": 0.0}
    for k in k_values:
        metrics[f"precision@{k}"] = 0.0
        metrics[f"recall@{k}"] = 0.0

    total_collisions = 0
    skipped_queries = 0

    for q in queries:
        query_text = q["query"]
        gold_chunk_ids = q["relevant_chunk_ids"]

        # Resolve gold spans from DB
        gold_spans: List[Tuple[str, int, int]] = []
        query_ok = True
        for gold_id in gold_chunk_ids:
            try:
                doc_id, cs, ce = resolve_gold_span_from_db(
                    gold_id, semantic_db_path
                )
                gold_spans.append((doc_id, cs, ce))
            except ValueError as e:
                logger.error(f"Skipping query {q['query_id']}: {e}")
                query_ok = False
                break

        if not query_ok:
            skipped_queries += 1
            continue

        # Retrieve
        retrieved = retriever.retrieve(query_text)

        retrieved_relevant: List[str] = []
        seen: set = set()
        for hit in retrieved:
            chunk_id = hit["chunk"].chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            ret_doc_id = hit["chunk"].doc_id
            ret_start = hit["chunk"].char_start
            ret_end = hit["chunk"].char_end

            # Check against all gold spans
            for g_doc_id, g_start, g_end in gold_spans:
                if is_span_relevant(
                    g_doc_id, g_start, g_end,
                    ret_doc_id, ret_start, ret_end,
                    overlap_threshold,
                ):
                    retrieved_relevant.append(chunk_id)
                    break

        # For scoring: treat each retrieved chunk as relevant/not-relevant
        # relevant_ids = list of chunk_ids that are span-relevant
        # We use the standard metric functions with the relevant set
        all_retrieved_ids = []
        seen2: set = set()
        for hit in retrieved:
            cid = hit["chunk"].chunk_id
            if cid not in seen2:
                seen2.add(cid)
                all_retrieved_ids.append(cid)

        metrics["mrr"] += calculate_mrr(all_retrieved_ids, retrieved_relevant)
        for k in k_values:
            metrics[f"precision@{k}"] += calculate_precision_at_k(
                all_retrieved_ids, retrieved_relevant, k
            )
            metrics[f"recall@{k}"] += calculate_recall_at_k(
                all_retrieved_ids, retrieved_relevant, k
            )

    effective_queries = len(queries) - skipped_queries
    if effective_queries > 0:
        metrics["mrr"] /= effective_queries
        for k in k_values:
            metrics[f"precision@{k}"] /= effective_queries
            metrics[f"recall@{k}"] /= effective_queries

    return {
        "metrics": metrics,
        "span_collision_count": total_collisions,
        "skipped_queries": skipped_queries,
        "effective_queries": effective_queries,
    }


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------


def compare_chunking_strategies(
    semantic_retriever: Any,
    fixed_retriever: Any,
    golden_set_path: str,
    output_path: str,
    mode: str = "id",
    semantic_db_path: str = "",
    semantic_eval_db_path: str = "",
    fixed_eval_db_path: str = "",
    overlap_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compare two retrieval strategies and save results.

    Args:
        semantic_retriever: The semantic chunking retriever.
        fixed_retriever: The fixed-512 chunking retriever.
        golden_set_path: Path to golden_set.json.
        output_path: Path to write results JSON.
        mode: "id" for ID matching, "span" for span overlap.
        semantic_db_path: Path to semantic metadata.db (for gold span lookup).
        semantic_eval_db_path: Path to semantic metadata.db (for span eval).
        fixed_eval_db_path: Path to fixed-512 metadata.db (for span eval).
        overlap_threshold: Minimum overlap ratio for span mode.

    Returns:
        The full comparison report dict.
    """
    if mode == "span":
        start = time.time()
        semantic_result = evaluate_retriever_span(
            semantic_retriever,
            golden_set_path,
            semantic_db_path,
            semantic_eval_db_path,
            overlap_threshold,
        )
        semantic_time = time.time() - start

        start = time.time()
        fixed_result = evaluate_retriever_span(
            fixed_retriever,
            golden_set_path,
            semantic_db_path,
            fixed_eval_db_path,
            overlap_threshold,
        )
        fixed_time = time.time() - start

        report = {
            "mode": "span",
            "overlap_threshold": overlap_threshold,
            "semantic_chunking": {
                "metrics": semantic_result["metrics"],
                "span_collision_count": semantic_result["span_collision_count"],
                "skipped_queries": semantic_result["skipped_queries"],
                "effective_queries": semantic_result["effective_queries"],
                "evaluation_time_seconds": semantic_time,
            },
            "fixed_512_chunking": {
                "metrics": fixed_result["metrics"],
                "span_collision_count": fixed_result["span_collision_count"],
                "skipped_queries": fixed_result["skipped_queries"],
                "effective_queries": fixed_result["effective_queries"],
                "evaluation_time_seconds": fixed_time,
            },
            "timestamp": time.time(),
        }
    else:
        start = time.time()
        semantic_results = evaluate_retriever(
            semantic_retriever, golden_set_path, k_values=[5]
        )
        semantic_time = time.time() - start

        start = time.time()
        fixed_results = evaluate_retriever(
            fixed_retriever, golden_set_path, k_values=[5]
        )
        fixed_time = time.time() - start

        report = {
            "mode": "id",
            "semantic_chunking": {
                "metrics": semantic_results,
                "evaluation_time_seconds": semantic_time,
            },
            "fixed_512_chunking": {
                "metrics": fixed_results,
                "evaluation_time_seconds": fixed_time,
            },
            "timestamp": time.time(),
        }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    return report


def create_golden_set_scaffold(output_path: str) -> None:
    """Create a scaffold golden set file."""
    scaffold = {
        "status": "PENDING_HUMAN_VERIFICATION",
        "version": "1.0",
        "queries": [
            {
                "query": "What is the governing law?",
                "relevant_chunk_ids": ["chunk_1"],
            }
        ],
    }
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(scaffold, f, indent=2)
