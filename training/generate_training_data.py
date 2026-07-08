import json
import random
import sqlite3
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

from loguru import logger
import hydra
from omegaconf import DictConfig

from src.retrieval.hybrid_retriever import HybridRetriever

def generate_queries_for_chunk(text, tokenizer, model, device, num_queries=2):
    """Generate synthetic queries for a given chunk of text using doc2query."""
    input_text = "macaw-answer: " + text  # Some t5 variants use prefixes, doc2query doesn't strictly need it but let's just pass text.
    # Actually doc2query/msmarco-t5-base-v1 just takes the text directly.
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        top_k=10,
        num_return_sequences=num_queries,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    queries = []
    for out in outputs:
        query = tokenizer.decode(out, skip_special_tokens=True)
        queries.append(query)
    return queries

from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_chunks", type=int, default=1500, help="Number of chunks to sample")
    parser.add_argument("--queries_per_chunk", type=int, default=2, help="Queries per chunk")
    parser.add_argument("--output_dir", type=str, default="data/training")
    
    args = parser.parse_args()
    
    cfg = OmegaConf.load("configs/config.yaml")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    logger.info(f"Using device: {device}")
    
    # 1. Generate Domain Triplets
    triplets_path = out_dir / "domain_triplets.jsonl"
    if not triplets_path.exists():
        logger.info("Initializing HybridRetriever for hard negatives...")
        retriever = HybridRetriever(
            faiss_path=cfg.indexer.faiss_index_path,
            bm25_path=cfg.indexer.bm25_index_path,
            metadata_db_path=cfg.indexer.metadata_db_path,
            manifest_path=str(Path(cfg.indexer.faiss_index_path).parent / "manifest.json"),
            model_name=cfg.indexer.embedding_model,
            rrf_k=cfg.retriever.rrf_k,
            top_k=10,
            device=device,
        )
        
        logger.info("Loading doc2query model...")
        d2q_name = "doc2query/msmarco-t5-base-v1"
        tokenizer = AutoTokenizer.from_pretrained(d2q_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(d2q_name).to(device)
        model.eval()
        
        # Load golden set and build excluded chunk IDs (including neighborhoods)
        golden_path = Path("evaluation/benchmarks/golden_set.json")
        excluded_chunk_ids = set()
        if golden_path.exists():
            with open(golden_path) as f:
                golden_data = json.load(f)
                
            with sqlite3.connect(cfg.indexer.metadata_db_path) as conn:
                cursor = conn.cursor()
                for q in golden_data.get("queries", []):
                    for cid in q.get("relevant_chunk_ids", []):
                        excluded_chunk_ids.add(cid)
                        # Find doc-span neighborhood (overlapping chunks)
                        cursor.execute("SELECT doc_id, char_start, char_end FROM chunks WHERE chunk_id = ?", (cid,))
                        res = cursor.fetchone()
                        if res:
                            doc_id, start, end = res
                            cursor.execute("SELECT chunk_id FROM chunks WHERE doc_id = ? AND char_start < ? AND char_end > ?", (doc_id, end, start))
                            neighbors = cursor.fetchall()
                            for (nid,) in neighbors:
                                excluded_chunk_ids.add(nid)
                                
            logger.info(f"Excluded {len(excluded_chunk_ids)} gold chunks (and neighborhoods) from query synthesis.")
        
        logger.info(f"Sampling {args.num_chunks} chunks from metadata.db...")
        with sqlite3.connect(cfg.indexer.metadata_db_path) as conn:
            cursor = conn.cursor()
            if excluded_chunk_ids:
                placeholders = ",".join("?" for _ in excluded_chunk_ids)
                query = f"SELECT chunk_id, text FROM chunks WHERE chunk_id NOT IN ({placeholders}) ORDER BY RANDOM() LIMIT ?"
                cursor.execute(query, tuple(excluded_chunk_ids) + (args.num_chunks,))
            else:
                cursor.execute("SELECT chunk_id, text FROM chunks ORDER BY RANDOM() LIMIT ?", (args.num_chunks,))
            chunks = cursor.fetchall()
            
        logger.info("Generating triplets...")
        with open(triplets_path, "w") as f:
            for chunk_id, text in tqdm(chunks):
                if len(text.split()) < 20:
                    continue  # skip very short chunks
                    
                queries = generate_queries_for_chunk(text, tokenizer, model, device, num_queries=args.queries_per_chunk)
                
                for query in queries:
                    # Retrieve hard negative
                    hits = retriever.retrieve(query)
                    hard_negative = None
                    for hit in hits:
                        if hit["chunk"].chunk_id != chunk_id:
                            hard_negative = hit["chunk"].text
                            break
                            
                    if hard_negative:
                        f.write(json.dumps({
                            "query": query,
                            "positive": text,
                            "negative": hard_negative,
                            "source_chunk_id": chunk_id
                        }) + "\n")
                        
        logger.info(f"Saved domain triplets to {triplets_path}")
    else:
        logger.info(f"{triplets_path} already exists. Skipping triplet generation.")
        
    # 2. Generate Query Classifier Data
    clf_path = out_dir / "query_classifier_data.jsonl"
    if not clf_path.exists():
        logger.info("Downloading HotpotQA and SQuAD for classifier training...")
        # HotpotQA for multi-hop
        hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
        # SQuAD for simple
        squad = load_dataset("rajpurkar/squad", split="train")
        
        classifier_data = []
        
        # Sample 3000 simple
        logger.info("Sampling 3000 simple queries from SQuAD...")
        squad_sample = squad.shuffle(seed=42).select(range(3000))
        for item in squad_sample:
            classifier_data.append({"query": item["question"], "label": "simple"})
            
        # Sample 1500 bridge and 1500 comparison from hotpot
        logger.info("Sampling 3000 multi-hop queries from HotpotQA...")
        bridge_count = 0
        comp_count = 0
        
        for item in hotpot.shuffle(seed=42):
            if item["type"] == "bridge" and bridge_count < 1500:
                classifier_data.append({"query": item["question"], "label": "multi-hop"})
                bridge_count += 1
            elif item["type"] == "comparison" and comp_count < 1500:
                classifier_data.append({"query": item["question"], "label": "comparative"})
                comp_count += 1
                
            if bridge_count >= 1500 and comp_count >= 1500:
                break
                
        random.shuffle(classifier_data)
        
        with open(clf_path, "w") as f:
            for item in classifier_data:
                f.write(json.dumps(item) + "\n")
                
        logger.info(f"Saved {len(classifier_data)} classifier examples to {clf_path}")
    else:
        logger.info(f"{clf_path} already exists. Skipping classifier data generation.")

if __name__ == "__main__":
    main()
