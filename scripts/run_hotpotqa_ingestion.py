"""Build a pooled index for HotpotQA distractor evaluation."""

import os
from pathlib import Path
from typing import List, Dict

import hydra
from omegaconf import DictConfig
from loguru import logger
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.ingestion.semantic_chunker import SemanticChunker
from src.ingestion.indexer import Indexer
from src.ingestion.document_parser import ParsedBlock

def get_hotpotqa_sample(num_samples: int = 200, seed: int = 42) -> List[Dict]:
    """Load HotpotQA distractor validation set and sample."""
    logger.info("Loading HotpotQA distractor validation dataset...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=seed).select(range(num_samples))
    
    samples = []
    for item in ds:
        titles = item["context"]["title"]
        sentences_lists = item["context"]["sentences"]
        gold_titles = set(item["supporting_facts"]["title"])
        
        paragraphs = []
        for title, sentences in zip(titles, sentences_lists):
            text = f"Title: {title}\n" + " ".join(sentences)
            paragraphs.append({"title": title, "text": text})
                
        samples.append({
            "id": item["id"],
            "question": item["question"],
            "paragraphs": paragraphs,
            "gold_titles": gold_titles
        })
    return samples

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    samples = get_hotpotqa_sample(num_samples=200, seed=42)
    
    out_dir = Path("data/hotpotqa_index")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    faiss_path = str(out_dir / "faiss.index")
    bm25_path = str(out_dir / "bm25.pkl")
    metadata_path = str(out_dir / "metadata.db")
    manifest_path = str(out_dir / "manifest.json")
    
    if Path(faiss_path).exists():
        logger.info(f"Removing existing index at {out_dir}")
        for p in out_dir.glob("*"):
            p.unlink()
            
    indexer = Indexer(
        faiss_path=faiss_path,
        bm25_path=bm25_path,
        metadata_db_path=metadata_path,
        manifest_path=manifest_path,
        embedding_dim=cfg.indexer.embedding_dim,
        embedding_model=cfg.indexer.embedding_model,
        chunker_config_hash="hotpotqa_pooled"
    )
    
    device = "mps" if (hasattr(os, "uname") and os.uname().machine == "arm64") else "cpu"
    
    chunker = SemanticChunker(
        model_name=cfg.chunker.model,
        similarity_threshold=cfg.chunker.similarity_threshold,
        min_tokens=cfg.chunker.min_tokens,
        max_tokens=cfg.chunker.max_tokens,
        batch_size=cfg.chunker.batch_size,
        device=device,
    )
    
    logger.info("Building pooled index...")
    
    processed_titles = set()
    total_chunks = 0
    
    for i, sample in enumerate(samples):
        for para in sample["paragraphs"]:
            title = para["title"]
            if title in processed_titles:
                continue
            processed_titles.add(title)
            
            # Use a dummy path for hashing, since it's from dataset
            doc_path = out_dir / "dummy" / f"{title.replace('/', '_')}.txt"
            doc_path.parent.mkdir(parents=True, exist_ok=True)
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
            chunks = chunker.chunk_document(title, [block])
            
            if chunks:
                texts = [c.text for c in chunks]
                embeddings = chunker.model.encode(
                    texts, batch_size=cfg.chunker.batch_size, normalize_embeddings=True, show_progress_bar=False
                )
                indexer.add_chunks(chunks, embeddings, doc_path)
                total_chunks += len(chunks)
                
        if (i + 1) % 20 == 0:
            logger.info(f"Processed {i+1}/200 questions. Chunks so far: {total_chunks}")
            
    logger.info(f"Pooled index complete. Total unique paragraphs: {len(processed_titles)}, Total chunks: {total_chunks}")

if __name__ == "__main__":
    main()
