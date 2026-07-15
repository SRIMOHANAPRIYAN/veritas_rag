import json
import sqlite3
import random
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import spacy

from omegaconf import OmegaConf
from src.generation.model_registry import registry
from src.generation.llm_client import LlamaClient

def get_excluded_docs(db_path: str, golden_path: str) -> set:
    excluded_docs = set()
    if not Path(golden_path).exists():
        return excluded_docs
        
    with open(golden_path) as f:
        golden_data = json.load(f)
        
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for q in golden_data.get("queries", []):
            for cid in q.get("relevant_chunk_ids", []):
                cursor.execute("SELECT doc_id FROM chunks WHERE chunk_id = ?", (cid,))
                res = cursor.fetchone()
                if res:
                    excluded_docs.add(res[0])
    return excluded_docs

async def generate_nli_data(num_samples: int = 1000):
    cfg = OmegaConf.load("configs/config.yaml")
    db_path = cfg.indexer.metadata_db_path
    
    excluded_docs = get_excluded_docs(db_path, "evaluation/benchmarks/golden_set.json")
    logger.info(f"Excluded {len(excluded_docs)} golden docs to prevent leakage.")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, text FROM chunks")
        all_chunks = cursor.fetchall()
        
    valid_chunks = [c for c in all_chunks if c[0] not in excluded_docs]
    
    logger.info("Loading spaCy to split sentences...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
        
    sentences = []
    for doc_id, text in valid_chunks:
        doc = nlp(text)
        for sent in doc.sents:
            sent_str = sent.text.strip()
            if 30 < len(sent_str) < 150: # reasonable length sentences
                sentences.append({"doc_id": doc_id, "text": sent_str})
                
    random.shuffle(sentences)
    
    llm = LlamaClient()
    
    target_per_class = num_samples // 3
    entailment_data = []
    contradiction_data = []
    neutral_data = []
    
    logger.info("Generating NLI pairs...")
    
    # Pre-select disjoint sets for neutral
    
    # We will do a batch generation using the LLM for paraphrases and negations
    idx = 0
    with tqdm(total=target_per_class * 2) as pbar:
        while idx < len(sentences) and (len(entailment_data) < target_per_class or len(contradiction_data) < target_per_class):
            item = sentences[idx]
            sent = item["text"]
            
            if len(entailment_data) < target_per_class:
                prompt = f"Paraphrase the following sentence. Do not add any extra text, just the paraphrase.\nSentence: {sent}\nParaphrase:"
                response = await llm.generate(prompt)
                para = response.strip()
                if para and para != sent:
                    entailment_data.append({"premise": sent, "hypothesis": para, "label": "entailment", "type": "synthetic_paraphrase"})
                    pbar.update(1)
                    
            if len(contradiction_data) < target_per_class:
                prompt = f"Rewrite the following sentence so that its meaning is strictly CONTRADICTED or negated. Do not add extra text.\nSentence: {sent}\nContradiction:"
                response = await llm.generate(prompt)
                contra = response.strip()
                if contra and contra != sent:
                    contradiction_data.append({"premise": sent, "hypothesis": contra, "label": "contradiction", "type": "synthetic_negation"})
                    pbar.update(1)
                    
            idx += 1
            
    # For neutral, we can just pair sentences from DIFFERENT docs
    logger.info("Generating Neutral pairs...")
    while len(neutral_data) < target_per_class:
        s1 = random.choice(sentences)
        s2 = random.choice(sentences)
        if s1["doc_id"] != s2["doc_id"]:
            neutral_data.append({"premise": s1["text"], "hypothesis": s2["text"], "label": "neutral", "type": "different_doc"})
            
    all_data = entailment_data + contradiction_data + neutral_data
    random.shuffle(all_data)
    
    out_dir = Path("data/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "domain_nli.jsonl"
    
    with open(out_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
            
    logger.info(f"Generated {len(all_data)} NLI pairs.")
    logger.info(f"Entailment: {len(entailment_data)}")
    logger.info(f"Contradiction: {len(contradiction_data)}")
    logger.info(f"Neutral: {len(neutral_data)}")
    logger.info(f"Saved to {out_path}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_nli_data(1000))
