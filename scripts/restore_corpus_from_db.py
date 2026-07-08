"""Restore corpus documents from the semantic metadata database.

Reconstructs full document text by concatenating all chunks per doc_id
(ordered by chunk_index) and writes them to data/reconstructed/{doc_id}.txt.
This provides a canonical text frame for span-overlap evaluation.
"""

import sqlite3
import sys
from pathlib import Path

from loguru import logger


def restore_corpus(
    metadata_db_path: str,
    output_dir: str,
) -> dict:
    """Restore all documents from metadata.db to individual text files.

    Args:
        metadata_db_path: Path to the semantic index metadata.db.
        output_dir: Directory to write reconstructed documents.

    Returns:
        Dict with keys: doc_count, total_chunks, output_dir.
    """
    db_path = Path(metadata_db_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        logger.error(f"metadata.db not found at {db_path}")
        sys.exit(1)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT doc_id FROM chunks ORDER BY doc_id"
        )
        doc_ids = [row[0] for row in cursor.fetchall()]

    logger.info(f"Found {len(doc_ids)} distinct documents in {db_path}")

    total_chunks = 0
    for doc_id in doc_ids:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC",
                (doc_id,),
            )
            chunk_texts = [row[0] for row in cursor.fetchall()]

        total_chunks += len(chunk_texts)
        full_text = "\n\n".join(chunk_texts)

        out_file = out_dir / f"{doc_id}.txt"
        out_file.write_text(full_text, encoding="utf-8")

    logger.info(
        f"Restored {len(doc_ids)} documents ({total_chunks} chunks) "
        f"to {out_dir}"
    )
    return {
        "doc_count": len(doc_ids),
        "total_chunks": total_chunks,
        "output_dir": str(out_dir),
    }


if __name__ == "__main__":
    import torch

    logger.info(f"Interpreter: {sys.prefix}")
    logger.info(f"Torch version: {torch.__version__}")

    result = restore_corpus(
        metadata_db_path="data/indexes/metadata.db",
        output_dir="data/reconstructed",
    )
    logger.info(f"Result: {result}")
