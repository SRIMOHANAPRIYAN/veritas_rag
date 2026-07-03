"""Calibrate semantic chunker similarity threshold."""

import os
from pathlib import Path
import numpy as np

import hydra
from omegaconf import DictConfig
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.ingestion.document_parser import DocumentParser


def get_sentence_similarities(model, texts):
    if not texts:
        return []
    embeddings = model.encode(texts, normalize_embeddings=True)
    if len(embeddings) < 2:
        return []
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return sims


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = (
        "mps" if (hasattr(os, "uname") and os.uname().machine == "arm64") else "cpu"
    )
    logger.info(f"Loading model {cfg.chunker.model} on {device}...")
    model = SentenceTransformer(cfg.chunker.model, device=device)

    import spacy

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000

    parser = DocumentParser()
    raw_dir = Path("data/raw")

    all_sims = []

    if raw_dir.exists():
        for file_path in raw_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [
                ".pdf",
                ".docx",
                ".html",
                ".txt",
            ]:
                logger.info(f"Parsing {file_path} for calibration...")
                try:
                    blocks = parser.parse(file_path)
                    for block in blocks:
                        if block.is_table:
                            continue
                        doc = nlp(block.text)
                        sentences = [
                            sent.text.strip() for sent in doc.sents if sent.text.strip()
                        ]
                        sims = get_sentence_similarities(model, sentences)
                        all_sims.extend(sims)
                except Exception as e:
                    logger.error(f"Failed {file_path}: {e}")

    if not all_sims:
        logger.warning("No data found for calibration. Return default 0.5")
        return

    all_sims = np.array(all_sims)
    q1 = np.percentile(all_sims, 25)
    median = np.median(all_sims)
    mean = np.mean(all_sims)

    logger.info(
        f"Calibration complete over {len(all_sims)} consecutive sentence pairs."
    )
    logger.info(f"Mean similarity:   {mean:.4f}")
    logger.info(f"Median similarity: {median:.4f}")
    logger.info(f"25th percentile:   {q1:.4f}")
    logger.info(f"Recommended threshold (approx 25th percentile): {q1:.4f}")


if __name__ == "__main__":
    main()
