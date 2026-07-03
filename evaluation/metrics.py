"""Evaluation metrics for RAG retrieval."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any


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


def evaluate_retriever(
    retriever, golden_set_path: str, k_values: List[int] = None
) -> Dict[str, float]:
    """Evaluate a retriever against a golden set."""
    if k_values is None:
        k_values = [10, 50]

    with open(golden_set_path, "r") as f:
        golden_set = json.load(f)

    queries = golden_set.get("queries", [])
    if not queries:
        return {}

    results = {"mrr": 0.0}
    for k in k_values:
        results[f"precision@{k}"] = 0.0
        results[f"recall@{k}"] = 0.0

    for q in queries:
        query_text = q["query"]
        relevant_docs = q["relevant_doc_ids"]

        retrieved = retriever.retrieve(query_text)

        # Deduplicate doc_ids in the order they appear
        retrieved_docs = []
        seen = set()
        for hit in retrieved:
            doc_id = hit["chunk"].doc_id
            if doc_id not in seen:
                seen.add(doc_id)
                retrieved_docs.append(doc_id)

        results["mrr"] += calculate_mrr(retrieved_docs, relevant_docs)
        for k in k_values:
            results[f"precision@{k}"] += calculate_precision_at_k(
                retrieved_docs, relevant_docs, k
            )
            results[f"recall@{k}"] += calculate_recall_at_k(
                retrieved_docs, relevant_docs, k
            )

    num_queries = len(queries)
    results["mrr"] /= num_queries
    for k in k_values:
        results[f"precision@{k}"] /= num_queries
        results[f"recall@{k}"] /= num_queries

    return results


def compare_chunking_strategies(
    semantic_retriever, fixed_retriever, golden_set_path: str, output_path: str
) -> Dict[str, Any]:
    """Compare two retrieval strategies and save results."""
    start = time.time()
    semantic_results = evaluate_retriever(semantic_retriever, golden_set_path)
    semantic_time = time.time() - start

    start = time.time()
    fixed_results = evaluate_retriever(fixed_retriever, golden_set_path)
    fixed_time = time.time() - start

    report = {
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


def create_golden_set_scaffold(output_path: str):
    """Create a scaffold golden set file."""
    scaffold = {
        "status": "PENDING_HUMAN_VERIFICATION",
        "version": "1.0",
        "queries": [
            {"query": "What is the governing law?", "relevant_doc_ids": ["doc_1"]}
        ],
    }
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(scaffold, f, indent=2)
