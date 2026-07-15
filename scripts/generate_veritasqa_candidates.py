#!/usr/bin/env python3
import json
import asyncio
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import random

from src.pipeline.query_pipeline import QueryPipeline

async def generate_candidates():
    golden_path = Path("evaluation/benchmarks/golden_set.json")
    if not golden_path.exists():
        logger.error(f"{golden_path} not found.")
        return

    with open(golden_path, "r") as f:
        golden_data = json.load(f)

    queries = [q["query"] for q in golden_data.get("queries", [])]
    
    logger.info(f"Loaded {len(queries)} queries from golden_set.json")
    
    pipeline = QueryPipeline()
    candidates = []
    
    # We will process queries and collect ClaimVerdicts
    for query in tqdm(queries, desc="Processing queries"):
        try:
            _, record = await pipeline.run(query)
            for claim in record.claims:
                candidates.append({
                    "query": query,
                    "claim_text": claim.claim_text,
                    "evidence_text": "Evidence ID: " + str(claim.evidence_chunk_id) + "\n\n(Retrieve from metadata.db or context for UI)",
                    "evidence_chunk_id": claim.evidence_chunk_id,
                    "model_verdict": claim.verdict,
                    "human_verdict": None
                })
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            
    logger.info(f"Generated {len(candidates)} total claims from the pipeline.")
    
    # For balancing, we could group by model_verdict and sample
    verdict_groups = {"ENTAILED": [], "CONTRADICTED": [], "BASELESS": [], "UNRESOLVED": []}
    for c in candidates:
        verdict_groups[c["model_verdict"]].append(c)
        
    for k, v in verdict_groups.items():
        logger.info(f"{k}: {len(v)} claims")
        
    # Attempt to build a balanced candidate pool of ~150-200 items for human review
    target_per_class = 50
    balanced_pool = []
    
    for k in ["ENTAILED", "CONTRADICTED", "BASELESS"]:
        group = verdict_groups[k]
        random.shuffle(group)
        balanced_pool.extend(group[:target_per_class])
        
    # Also add some unresolved if any
    unresolved = verdict_groups.get("UNRESOLVED", [])
    if unresolved:
        random.shuffle(unresolved)
        balanced_pool.extend(unresolved[:20])
        
    random.shuffle(balanced_pool)
    
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "veritasqa_candidates.jsonl"
    
    with open(out_path, "w") as f:
        for item in balanced_pool:
            f.write(json.dumps(item) + "\n")
            
    logger.info(f"Saved {len(balanced_pool)} candidates to {out_path} for human review.")

if __name__ == "__main__":
    asyncio.run(generate_candidates())
