"""Part B — Query Classifier v2 sanity check.

1. Print the model's id2label mapping.
2. Run a fresh 15-query probe (5 simple, 5 multi-hop, 5 comparative).
3. Evaluate WITH keyword fallback active.
4. Report accuracy and per-query predictions.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf

# ── interpreter guard ──
if "anaconda" in sys.prefix.lower():
    logger.error(f"Running under Anaconda ({sys.prefix}). Use the project venv.")
    sys.exit(1)

logger.info(f"Interpreter: {sys.prefix}")

DEVICE = "mps" if (hasattr(os, "uname") and os.uname().machine == "arm64") else "cpu"

from src.retrieval.query_classifier import QueryClassifier

MODEL_PATH = "models/query_classifier_v2/"
PROBE_PATH = "evaluation/benchmarks/classifier_probe_v2_queries.json"
OUTPUT_PATH = "evaluation/benchmarks/classifier_probe_v2.json"

def main():
    cfg = OmegaConf.load("configs/config.yaml")
    fallback_keywords = OmegaConf.to_container(cfg.classifier.fallback_keywords, resolve=True) if "classifier" in cfg and "fallback_keywords" in cfg.classifier else {}

    # ── 1. Load model & print id2label ──
    logger.info(f"Loading classifier from {MODEL_PATH}")
    classifier = QueryClassifier(MODEL_PATH, device=DEVICE, fallback_keywords=fallback_keywords)

    if classifier.model is None:
        logger.error("Classifier model failed to load! Cannot proceed.")
        sys.exit(1)

    id2label = classifier.model.config.id2label
    label2id = classifier.model.config.label2id
    logger.info(f"id2label mapping: {json.dumps(id2label, indent=2)}")
    logger.info(f"label2id mapping: {json.dumps(label2id, indent=2)}")

    # ── 2. Load Probe Queries ──
    with open(PROBE_PATH) as f:
        probe_queries = json.load(f)

    # ── 3. Run probe ──
    results = []
    correct = 0
    total = len(probe_queries)

    for pq in probe_queries:
        query = pq["query"]
        expected = pq["expected"]

        # Get raw logits + softmax for confidence
        inputs = classifier.tokenizer(query, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = classifier.model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0)
        
        # Get final label via the official classify method (which includes fallback logic)
        final_label = classifier.classify(query)
        
        # We also record what the raw model predicted before fallback
        pred_idx = logits.argmax().item()
        raw_model_label = id2label.get(str(pred_idx), id2label.get(pred_idx, f"UNKNOWN_{pred_idx}"))
        confidence = probs[pred_idx].item()

        is_correct = final_label == expected
        if is_correct:
            correct += 1

        results.append({
            "query": query,
            "expected": expected,
            "predicted": final_label,
            "raw_model_predicted": raw_model_label,
            "fallback_applied": final_label != raw_model_label,
            "confidence": round(confidence, 4),
            "correct": is_correct,
            "logits": {id2label.get(str(i), id2label.get(i, f"label_{i}")): round(logits[i].item(), 4) for i in range(len(logits))},
        })

    accuracy = correct / total

    # ── 4. Print report ──
    logger.info("=" * 80)
    logger.info("CLASSIFIER PROBE v2 RESULTS")
    logger.info("=" * 80)

    for i, r in enumerate(results, 1):
        status = "✓" if r["correct"] else "✗"
        fallback_flag = " [FALLBACK]" if r["fallback_applied"] else ""
        logger.info(
            f"  {status} [{i:2d}/15] expected={r['expected']:12s}  "
            f"predicted={r['predicted']:12s}{fallback_flag}  conf={r['confidence']:.4f}  "
            f"query=\"{r['query'][:60]}...\""
        )

    logger.info(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

    if accuracy < 12 / 15:
        logger.warning(f"Accuracy {accuracy:.2%} < 80% (12/15) pass bar.")
    else:
        logger.info(f"Accuracy {accuracy:.2%} >= 80% (12/15) pass bar. PASS.")

    # ── 5. Save results ──
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

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Probe report written to {out_path}")


if __name__ == "__main__":
    main()
