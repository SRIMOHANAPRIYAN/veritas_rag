"""Part B — Query Classifier sanity check.

1. Print the model's id2label mapping.
2. Run a 15-query probe (5 simple, 5 multi-hop, 5 comparative).
3. Report accuracy and per-query predictions.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger

# ── interpreter guard ──
if "anaconda" in sys.prefix.lower():
    logger.error(f"Running under Anaconda ({sys.prefix}). Use the project venv.")
    sys.exit(1)

logger.info(f"Interpreter: {sys.prefix}")

DEVICE = "mps" if (hasattr(os, "uname") and os.uname().machine == "arm64") else "cpu"

from src.retrieval.query_classifier import QueryClassifier

MODEL_PATH = "models/query_classifier/"

# ═══════════════════════════════════════════════════════════════════════════
# 15-QUERY PROBE SET (fresh, mixed domains)
# ═══════════════════════════════════════════════════════════════════════════
PROBE_QUERIES = [
    # ── 5 SIMPLE (single-hop, factoid) ──
    {"query": "What is the termination date of the agreement?", "expected": "simple"},
    {"query": "Who is the licensor under this contract?", "expected": "simple"},
    {"query": "What percentage royalty does the supplier receive?", "expected": "simple"},
    {"query": "Where is the registered office of the company?", "expected": "simple"},
    {"query": "What is the governing law for this franchise agreement?", "expected": "simple"},

    # ── 5 MULTI-HOP (require bridging across 2+ facts) ──
    {"query": "Who is the CEO of the company that signed the distribution agreement with Acme Corp?", "expected": "multi-hop"},
    {"query": "What is the address of the supplier who provides components mentioned in the maintenance clause?", "expected": "multi-hop"},
    {"query": "Which subsidiary executed the IP agreement, and what parent company owns it?", "expected": "multi-hop"},
    {"query": "What warranty terms apply to the product referenced in the supply agreement between the two joint-venture partners?", "expected": "multi-hop"},
    {"query": "Under the hosting agreement, which data center provider is responsible for the uptime SLA, and what penalties apply?", "expected": "multi-hop"},

    # ── 5 COMPARATIVE ──
    {"query": "How does the indemnification clause in the 2019 agreement differ from the 2020 amendment?", "expected": "comparative"},
    {"query": "Compare the revenue-sharing terms between the distributor agreement and the reseller agreement.", "expected": "comparative"},
    {"query": "What are the differences in termination rights between the licensor and the licensee?", "expected": "comparative"},
    {"query": "Which agreement offers a longer exclusivity period, the strategic alliance or the joint venture?", "expected": "comparative"},
    {"query": "What was the revenue in 2019 and how did it compare to the company acquired in 2020?", "expected": "comparative"},
]


def main():
    # ── 1. Load model & print id2label ──
    logger.info(f"Loading classifier from {MODEL_PATH}")
    classifier = QueryClassifier(MODEL_PATH, device=DEVICE)

    if classifier.model is None:
        logger.error("Classifier model failed to load! Cannot proceed.")
        sys.exit(1)

    id2label = classifier.model.config.id2label
    label2id = classifier.model.config.label2id
    logger.info(f"id2label mapping: {json.dumps(id2label, indent=2)}")
    logger.info(f"label2id mapping: {json.dumps(label2id, indent=2)}")

    # ── 2. Run probe ──
    results = []
    correct = 0
    total = len(PROBE_QUERIES)

    for pq in PROBE_QUERIES:
        query = pq["query"]
        expected = pq["expected"]

        # Get raw logits + softmax for confidence
        inputs = classifier.tokenizer(query, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = classifier.model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0)
        pred_idx = logits.argmax().item()
        pred_label = id2label.get(str(pred_idx), id2label.get(pred_idx, f"UNKNOWN_{pred_idx}"))
        confidence = probs[pred_idx].item()

        is_correct = pred_label == expected
        if is_correct:
            correct += 1

        results.append({
            "query": query,
            "expected": expected,
            "predicted": pred_label,
            "confidence": round(confidence, 4),
            "correct": is_correct,
            "logits": {id2label.get(str(i), id2label.get(i, f"label_{i}")): round(logits[i].item(), 4) for i in range(len(logits))},
        })

    accuracy = correct / total

    # ── 3. Print report ──
    logger.info("=" * 80)
    logger.info("CLASSIFIER PROBE RESULTS")
    logger.info("=" * 80)

    for i, r in enumerate(results, 1):
        status = "✓" if r["correct"] else "✗"
        logger.info(
            f"  {status} [{i:2d}/15] expected={r['expected']:12s}  "
            f"predicted={r['predicted']:12s}  conf={r['confidence']:.4f}  "
            f"query=\"{r['query'][:70]}...\""
        )

    logger.info(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

    if accuracy < 10 / 15:
        logger.warning(
            f"Accuracy {accuracy:.2%} < 66.7% threshold. "
            "DIAGNOSIS REQUIRED before any retraining."
        )

    # ── 4. Save results ──
    report = {
        "model_path": MODEL_PATH,
        "id2label": id2label,
        "label2id": label2id,
        "probe_accuracy": round(accuracy, 4),
        "probe_correct": correct,
        "probe_total": total,
        "probe_results": results,
        "timestamp": time.time(),
    }

    out_path = Path("evaluation/benchmarks/classifier_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Probe report written to {out_path}")


if __name__ == "__main__":
    main()
