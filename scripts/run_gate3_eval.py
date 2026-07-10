"""Gate 3 Evaluation on HotpotQA distractor setting."""

import os
import time
import json
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.indexer import Indexer
from src.ingestion.document_parser import ParsedBlock
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agents.iterative_retrieval_agent import IterativeRetrievalAgent
from src.generation.model_registry import registry
from src.common.device import DEVICE

def get_hotpotqa_sample(num_samples: int = 200, seed: int = 42) -> List[Dict]:
    """Load HotpotQA distractor validation set and sample."""
    logger.info("Loading HotpotQA distractor validation dataset...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    ds = ds.shuffle(seed=seed).select(range(num_samples))
    
    samples = []
    for item in ds:
        # Context is a dict of titles and sentences
        titles = item["context"]["title"]
        sentences_lists = item["context"]["sentences"]
        
        # Gold paragraphs are in supporting_facts
        gold_titles = set(item["supporting_facts"]["title"])
        
        paragraphs = []
        gold_indices = []
        for i, (title, sentences) in enumerate(zip(titles, sentences_lists)):
            text = f"Title: {title}\n" + " ".join(sentences)
            paragraphs.append({"title": title, "text": text})
            if title in gold_titles:
                gold_indices.append(i)
                
        samples.append({
            "id": item["id"],
            "question": item["question"],
            "paragraphs": paragraphs,
            "gold_titles": gold_titles
        })
    return samples

async def evaluate_question(
    sample: Dict, 
    cfg: DictConfig, 
    chunker: SemanticChunker,
    temp_dir: str
) -> Dict[str, Any]:
    """Evaluate a single question: index 10 paras, retrieve single-shot vs multi-hop."""
    
    # 1. Index the 10 paragraphs in a temporary index
    sample_dir = Path(temp_dir) / sample['id']
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    faiss_path = str(sample_dir / "faiss.index")
    bm25_path = str(sample_dir / "bm25.pkl")
    metadata_path = str(sample_dir / "metadata.db")
    manifest_path = str(sample_dir / "manifest.json")
    
    indexer = Indexer(
        faiss_path=faiss_path,
        bm25_path=bm25_path,
        metadata_db_path=metadata_path,
        manifest_path=manifest_path,
        embedding_dim=cfg.indexer.embedding_dim,
        embedding_model=cfg.indexer.embedding_model,
        chunker_config_hash="hotpotqa_eval"
    )
    
    # Add paragraphs to index
    all_chunks = []
    for i, para in enumerate(sample["paragraphs"]):
        # Sanitize title for filename
        safe_title = "".join([c if c.isalnum() else "_" for c in para["title"]])
        doc_id = para["title"] # Keep original for recall comparison
        
        # Write to temporary file so Indexer can hash it
        doc_path = sample_dir / f"{safe_title}.txt"
        doc_path.write_text(para["text"], encoding="utf-8")
        
        block = ParsedBlock(
            text=para["text"],
            page=1,
            heading_path=[],
            is_table=False,
            source_path=str(doc_path),
            char_start=0,
            char_end=len(para["text"])
        )
        chunks = chunker.chunk_document(doc_id, [block])
        
        if chunks:
            texts = [c.text for c in chunks]
            embeddings = chunker.model.encode(
                texts, batch_size=cfg.chunker.batch_size, normalize_embeddings=True, show_progress_bar=False
            )
            indexer.add_chunks(chunks, embeddings, doc_path)
            all_chunks.extend(chunks)
            
    # 2. Override config dynamically for this query so HybridRetriever uses this index
    OmegaConf.update(registry.cfg, "indexer.faiss_index_path", faiss_path)
    OmegaConf.update(registry.cfg, "indexer.bm25_index_path", bm25_path)
    OmegaConf.update(registry.cfg, "indexer.metadata_db_path", metadata_path)
    
    # Force retriever to reload
    registry._hybrid_retriever = None
    registry.get_hybrid_retriever()
    
    retriever = registry.get_hybrid_retriever()
    reranker = registry.get_reranker()
    agent = IterativeRetrievalAgent(max_iterations=3)
    
    question = sample["question"]
    gold_titles = sample["gold_titles"]
    
    # --- Single-shot Baseline ---
    single_candidates = retriever.retrieve(question)
    single_reranked = reranker.rerank(question, single_candidates)
    
    # Calculate Recall@5 for single-shot
    # A gold paragraph is successfully retrieved if any of its chunks are in the top 5
    single_retrieved_docs = {c["chunk"].doc_id for c in single_reranked[:5]}
    single_recall = len(single_retrieved_docs.intersection(gold_titles)) / len(gold_titles) if gold_titles else 0
    
    # --- Multi-hop Agent ---
    agent_chunks = await agent.run(question)
    agent_retrieved_docs = {c["chunk"].doc_id for c in agent_chunks[:5]}
    agent_recall = len(agent_retrieved_docs.intersection(gold_titles)) / len(gold_titles) if gold_titles else 0
    
    return {
        "id": sample["id"],
        "single_recall": single_recall,
        "agent_recall": agent_recall
    }

async def run_eval(cfg: DictConfig):
    samples = get_hotpotqa_sample(num_samples=200, seed=42)
    
    device = "mps" if (hasattr(os, "uname") and os.uname().machine == "arm64") else "cpu"
    
    chunker = SemanticChunker(
        model_name=cfg.chunker.model,
        similarity_threshold=cfg.chunker.similarity_threshold,
        min_tokens=cfg.chunker.min_tokens,
        max_tokens=cfg.chunker.max_tokens,
        batch_size=cfg.chunker.batch_size,
        device=device,
    )
    
    results = []
    
    # Run evaluations sequentially (to avoid NLI/LLM concurrency issues or high memory)
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating {i+1}/{len(samples)}: {sample['question']}")
            res = await evaluate_question(sample, cfg, chunker, temp_dir)
            results.append(res)
            
            # Print running average
            avg_single = np.mean([r["single_recall"] for r in results])
            avg_agent = np.mean([r["agent_recall"] for r in results])
            logger.info(f"Running Avg MSR -> Single: {avg_single:.4f} | Agent: {avg_agent:.4f}")
            
    # Final metrics
    avg_single = float(np.mean([r["single_recall"] for r in results]))
    avg_agent = float(np.mean([r["agent_recall"] for r in results]))
    
    logger.info("="*50)
    logger.info(f"Gate 3 Evaluation Complete (200 samples)")
    logger.info(f"Single-Shot Baseline MSR: {avg_single:.4f}")
    logger.info(f"Multi-Hop Agent MSR:      {avg_agent:.4f}")
    logger.info("="*50)
    
    out_file = Path("evaluation/benchmarks/results_phase3.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "metrics": {
                "single_shot_msr": avg_single,
                "multi_hop_agent_msr": avg_agent
            },
            "samples": results
        }, f, indent=2)
    logger.info(f"Saved results to {out_file}")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Setup LLM n_ctx in OmegaConf since it's required
    if "generation" not in cfg:
        cfg.generation = {}
    if "n_ctx" not in cfg.generation:
        cfg.generation.n_ctx = 8192
        
    asyncio.run(run_eval(cfg))

if __name__ == "__main__":
    main()
