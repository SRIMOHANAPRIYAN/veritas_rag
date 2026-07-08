"""Tests for data leakage in training data generation."""

import json
from pathlib import Path


def _load_gold_chunk_ids() -> set:
    """Load all gold chunk IDs from golden_set.json."""
    golden_path = Path("evaluation/benchmarks/golden_set.json")
    if not golden_path.exists():
        return set()

    with open(golden_path) as f:
        golden_data = json.load(f)

    gold_ids = set()
    for q in golden_data.get("queries", []):
        for cid in q.get("relevant_chunk_ids", []):
            gold_ids.add(cid)
    return gold_ids


def test_no_data_leakage_in_training_triplets():
    """No generated triplet's source_chunk_id should appear in the golden set."""
    triplets_path = Path("data/training/domain_triplets.jsonl")
    gold_chunk_ids = _load_gold_chunk_ids()

    if not triplets_path.exists() or not gold_chunk_ids:
        return

    leaked = []
    with open(triplets_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            source_id = item.get("source_chunk_id")
            if source_id and source_id in gold_chunk_ids:
                leaked.append(source_id)

    assert len(leaked) == 0, (
        f"Found {len(leaked)} leaked gold chunks in domain_triplets.jsonl!"
    )


def test_no_data_leakage_in_classifier_data_v2():
    """No v2 classifier query's source_chunk_ids should appear in golden set."""
    clf_path = Path("data/training/query_classifier_data_v2.jsonl")
    gold_chunk_ids = _load_gold_chunk_ids()

    if not clf_path.exists() or not gold_chunk_ids:
        return

    leaked = []
    with open(clf_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            source_ids = item.get("source_chunk_ids", [])
            for sid in source_ids:
                if sid in gold_chunk_ids:
                    leaked.append(sid)

    assert len(leaked) == 0, (
        f"Found {len(leaked)} leaked gold chunks in "
        f"query_classifier_data_v2.jsonl!"
    )
