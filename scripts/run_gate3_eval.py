"""Gate 3 Evaluation on HotpotQA distractor setting (pooled index).

Resilient version:
- RESUME: skips question IDs already in results_phase3.jsonl (append mode).
- SKIP LIST: skips IDs in gate3_skip.txt (poison questions that hang llama.cpp),
  writing a null/skipped record so the run can complete.
- HEARTBEAT: writes the in-flight question ID to gate3_current.txt before each
  question so an external watchdog can identify a wedged question.
Run via the watchdog wrapper: bash scripts/run_gate3_watchdog.sh
"""

import json
import asyncio
import gc
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from datasets import load_dataset

from src.agents.iterative_retrieval_agent import IterativeRetrievalAgent
from src.generation.model_registry import registry

BENCH = Path("evaluation/benchmarks")
JSONL_FILE = BENCH / "results_phase3.jsonl"
SUMMARY_FILE = BENCH / "results_phase3.json"
SKIP_FILE = BENCH / "gate3_skip.txt"
CURRENT_FILE = BENCH / "gate3_current.txt"
NUM_SAMPLES = 200
SEED = 42


def get_hotpotqa_sample(num_samples: int = NUM_SAMPLES, seed: int = SEED) -> List[Dict]:
    """Load HotpotQA distractor validation set and sample deterministically."""
    logger.info("Loading HotpotQA distractor validation dataset...")
    ds = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True
    )
    ds = ds.shuffle(seed=seed).select(range(num_samples))
    samples = []
    for item in ds:
        gold_titles = set(item["supporting_facts"]["title"])
        samples.append(
            {"id": item["id"], "question": item["question"], "gold_titles": gold_titles}
        )
    return samples


async def evaluate_question(sample: Dict, cfg: DictConfig) -> Dict[str, Any]:
    """Single-shot vs multi-hop agent recall@5 against the pre-built pooled index."""
    OmegaConf.update(registry.cfg, "indexer.faiss_index_path", "data/hotpotqa_index/faiss.index")
    OmegaConf.update(registry.cfg, "indexer.bm25_index_path", "data/hotpotqa_index/bm25.pkl")
    OmegaConf.update(registry.cfg, "indexer.metadata_db_path", "data/hotpotqa_index/metadata.db")
    registry._hybrid_retriever = None
    retriever = registry.get_hybrid_retriever()
    reranker = registry.get_reranker()
    agent = IterativeRetrievalAgent(
        max_iterations=3, max_sub_questions=cfg.generation.get("max_sub_questions", 3)
    )

    question = sample["question"]
    gold = sample["gold_titles"]

    single = reranker.rerank(question, retriever.retrieve(question))
    single_docs = {c["chunk"].doc_id for c in single[:5]}
    single_recall = len(single_docs & gold) / len(gold) if gold else 0.0

    agent_chunks = await agent.run(question)
    agent_docs = {c["chunk"].doc_id for c in agent_chunks[:5]}
    agent_recall = len(agent_docs & gold) / len(gold) if gold else 0.0

    return {"id": sample["id"], "single_recall": single_recall, "agent_recall": agent_recall}


def _load_done_ids() -> set:
    if not JSONL_FILE.exists():
        return set()
    done = set()
    for line in JSONL_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["id"])
        except Exception:
            pass
    return done


def _load_skip_ids() -> set:
    if not SKIP_FILE.exists():
        return set()
    return {ln.strip() for ln in SKIP_FILE.read_text().splitlines() if ln.strip()}


async def run_eval(cfg: DictConfig):
    samples = get_hotpotqa_sample()
    done = _load_done_ids()
    skip = _load_skip_ids()
    logger.info(f"Resume: {len(done)} done, {len(skip)} skip-listed, {len(samples)} total.")

    with open(JSONL_FILE, "a") as f:  # APPEND — never wipe prior progress
        for i, sample in enumerate(samples):
            qid = sample["id"]
            if qid in done:
                continue
            if qid in skip:
                logger.warning(f"SKIP-LISTED (poison) {qid} — writing null record.")
                f.write(json.dumps({"id": qid, "single_recall": None,
                                    "agent_recall": None, "skipped": True}) + "\n")
                f.flush()
                done.add(qid)
                continue

            CURRENT_FILE.write_text(qid)  # heartbeat for the watchdog
            logger.info(f"[{i+1}/{len(samples)}] {qid}: {sample['question'][:80]}")
            res = await evaluate_question(sample, cfg)
            f.write(json.dumps(res) + "\n")
            f.flush()
            done.add(qid)

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    _write_summary()


def _write_summary():
    rows = [json.loads(l) for l in JSONL_FILE.read_text().splitlines() if l.strip()]
    scored = [r for r in rows if r.get("single_recall") is not None]
    skipped = [r["id"] for r in rows if r.get("skipped")]
    single = float(np.mean([r["single_recall"] for r in scored])) if scored else 0.0
    agent = float(np.mean([r["agent_recall"] for r in scored])) if scored else 0.0
    hard = [r for r in scored if r["single_recall"] < 1.0]
    hs = float(np.mean([r["single_recall"] for r in hard])) if hard else 0.0
    ha = float(np.mean([r["agent_recall"] for r in hard])) if hard else 0.0
    summary = {
        "metrics": {"single_shot_msr": single, "multi_hop_agent_msr": agent},
        "hard_subset": {"n": len(hard), "single": hs, "agent": ha},
        "scored": len(scored), "skipped_ids": skipped,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    logger.info("=" * 50)
    logger.info(f"Scored {len(scored)}/{NUM_SAMPLES} | skipped {len(skipped)}: {skipped}")
    logger.info(f"Overall  -> Single {single:.4f} | Agent {agent:.4f}")
    logger.info(f"Hard(n={len(hard)}) -> Single {hs:.4f} | Agent {ha:.4f}")
    logger.info("=" * 50)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if "generation" not in cfg:
        cfg.generation = {}
    if "n_ctx" not in cfg.generation:
        cfg.generation.n_ctx = 8192
    asyncio.run(run_eval(cfg))


if __name__ == "__main__":
    main()
