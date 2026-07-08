"""Generate legal-domain classifier training data (v2).

Creates domain-specific multi-hop and comparative queries using
templates over chunk pairs from the CUAD corpus. Mixed 50:50 with
existing SQuAD/HotpotQA general-domain data.

Standalone: python training/generate_legal_classifier_data.py
"""

import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Set

from loguru import logger
from omegaconf import OmegaConf

# -- Templates --
COMPARATIVE_TEMPLATES = [
    "How does the {topic} in {doc_a} differ from {doc_b}?",
    "Compare the {topic} clause between {doc_a} and {doc_b}.",
    "What are the differences in {topic} terms between {doc_a} and {doc_b}?",
    "Which agreement has more favorable {topic} provisions, {doc_a} or {doc_b}?",
    "Contrast the {topic} obligations in {doc_a} versus {doc_b}.",
    "Between {doc_a} and {doc_b}, which offers stricter {topic} requirements?",
    "How do the {topic} conditions in {doc_a} compare with those in {doc_b}?",
]

MULTI_HOP_TEMPLATES = [
    "Under {doc}, what {topic_a} applies to the party responsible for {topic_b}?",
    "In {doc}, who is obligated under the {topic_a} clause, and what {topic_b} terms bind them?",
    "What {topic_a} rights does the party with {topic_b} obligations have under {doc}?",
    "In {doc}, how does the {topic_a} provision interact with the {topic_b} section?",
    "Under {doc}, what happens to {topic_a} if {topic_b} conditions are breached?",
    "In {doc}, which party's {topic_a} duties are affected by the {topic_b} clause?",
]

# -- Legal clause topics --
CLAUSE_TOPICS = [
    "termination",
    "indemnification",
    "liability",
    "warranty",
    "confidentiality",
    "intellectual property",
    "payment",
    "governing law",
    "force majeure",
    "non-compete",
    "exclusivity",
    "assignment",
    "notice",
    "dispute resolution",
    "renewal",
    "insurance",
    "compliance",
    "data protection",
    "representations",
    "covenants",
]


def load_excluded_chunk_ids(db_path: str) -> Set[str]:
    """Load gold chunk IDs and their neighborhoods to exclude."""
    golden_path = Path("evaluation/benchmarks/golden_set.json")
    excluded: Set[str] = set()
    if not golden_path.exists():
        return excluded

    with open(golden_path) as f:
        golden = json.load(f)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for q in golden.get("queries", []):
            for cid in q.get("relevant_chunk_ids", []):
                excluded.add(cid)
                cursor.execute(
                    "SELECT doc_id, char_start, char_end FROM chunks "
                    "WHERE chunk_id = ?",
                    (cid,),
                )
                res = cursor.fetchone()
                if res:
                    doc_id, start, end = res
                    cursor.execute(
                        "SELECT chunk_id FROM chunks "
                        "WHERE doc_id = ? AND char_start < ? AND char_end > ?",
                        (doc_id, end, start),
                    )
                    for (nid,) in cursor.fetchall():
                        excluded.add(nid)

    return excluded


def extract_doc_short_name(doc_id: str) -> str:
    """Extract a readable short name from a doc_id."""
    name = doc_id.replace("_", " ")
    parts = name.split("-")
    if len(parts) > 1:
        return parts[0].strip()[:60]
    return name[:60]


def generate_comparative_queries(
    chunks_by_doc: Dict[str, List[dict]],
    excluded: Set[str],
    num_queries: int = 1500,
) -> List[dict]:
    """Generate comparative queries from cross-doc same-topic pairs."""
    rng = random.Random(42)
    queries: List[dict] = []
    doc_ids = list(chunks_by_doc.keys())

    if len(doc_ids) < 2:
        logger.warning("Not enough docs for comparative queries")
        return queries

    attempts = 0
    max_attempts = num_queries * 10

    while len(queries) < num_queries and attempts < max_attempts:
        attempts += 1
        doc_a, doc_b = rng.sample(doc_ids, 2)
        topic = rng.choice(CLAUSE_TOPICS)
        template = rng.choice(COMPARATIVE_TEMPLATES)

        chunks_a = [c for c in chunks_by_doc[doc_a] if c["chunk_id"] not in excluded]
        chunks_b = [c for c in chunks_by_doc[doc_b] if c["chunk_id"] not in excluded]
        if not chunks_a or not chunks_b:
            continue

        src_a = rng.choice(chunks_a)
        src_b = rng.choice(chunks_b)

        name_a = extract_doc_short_name(doc_a)
        name_b = extract_doc_short_name(doc_b)

        query = template.format(topic=topic, doc_a=name_a, doc_b=name_b)
        queries.append(
            {
                "query": query,
                "label": "comparative",
                "source_chunk_ids": [src_a["chunk_id"], src_b["chunk_id"]],
            }
        )

    logger.info(f"Generated {len(queries)} comparative queries")
    return queries


def generate_multi_hop_queries(
    chunks_by_doc: Dict[str, List[dict]],
    excluded: Set[str],
    num_queries: int = 1500,
) -> List[dict]:
    """Generate multi-hop queries from within-doc cross-section pairs."""
    rng = random.Random(43)
    queries: List[dict] = []

    eligible_docs = [
        doc_id
        for doc_id, chunks in chunks_by_doc.items()
        if len([c for c in chunks if c["chunk_id"] not in excluded]) >= 3
    ]

    if not eligible_docs:
        logger.warning("Not enough eligible docs for multi-hop queries")
        return queries

    attempts = 0
    max_attempts = num_queries * 10

    while len(queries) < num_queries and attempts < max_attempts:
        attempts += 1
        doc_id = rng.choice(eligible_docs)
        eligible_chunks = [
            c for c in chunks_by_doc[doc_id] if c["chunk_id"] not in excluded
        ]
        if len(eligible_chunks) < 2:
            continue

        chunk_a, chunk_b = rng.sample(eligible_chunks, 2)
        topic_a, topic_b = rng.sample(CLAUSE_TOPICS, 2)
        template = rng.choice(MULTI_HOP_TEMPLATES)

        doc_name = extract_doc_short_name(doc_id)
        query = template.format(doc=doc_name, topic_a=topic_a, topic_b=topic_b)
        queries.append(
            {
                "query": query,
                "label": "multi-hop",
                "source_chunk_ids": [
                    chunk_a["chunk_id"],
                    chunk_b["chunk_id"],
                ],
            }
        )

    logger.info(f"Generated {len(queries)} multi-hop queries")
    return queries


def load_existing_general_data(data_path: Path, sample_size: int = 3000) -> List[dict]:
    """Load existing SQuAD/HotpotQA classifier data."""
    if not data_path.exists():
        logger.warning(f"Existing classifier data not found at {data_path}")
        return []

    all_data: List[dict] = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

    rng = random.Random(42)
    if len(all_data) > sample_size:
        all_data = rng.sample(all_data, sample_size)

    logger.info(f"Loaded {len(all_data)} general-domain samples from {data_path}")
    return all_data


def main() -> None:
    """Generate legal-domain classifier training data."""
    cfg = OmegaConf.load("configs/config.yaml")
    db_path: str = cfg.indexer.metadata_db_path

    out_path = Path("data/training/query_classifier_data_v2.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load excluded chunks
    excluded = load_excluded_chunk_ids(db_path)
    logger.info(f"Excluded {len(excluded)} gold chunks (and neighborhoods)")

    # Load all chunks grouped by doc
    chunks_by_doc: Dict[str, List[dict]] = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, doc_id, text FROM chunks")
        for chunk_id, doc_id, text in cursor.fetchall():
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(
                {"chunk_id": chunk_id, "doc_id": doc_id, "text": text}
            )

    logger.info(
        f"Loaded chunks from {len(chunks_by_doc)} docs "
        f"({sum(len(v) for v in chunks_by_doc.values())} total chunks)"
    )

    # Generate legal-domain queries
    comp_queries = generate_comparative_queries(chunks_by_doc, excluded, 1500)
    mhop_queries = generate_multi_hop_queries(chunks_by_doc, excluded, 1500)

    domain_data = comp_queries + mhop_queries
    logger.info(f"Total domain queries: {len(domain_data)}")

    # Load existing general data (~50:50 mix)
    general_data = load_existing_general_data(
        Path("data/training/query_classifier_data.jsonl"),
        sample_size=len(domain_data),
    )

    # Combine and shuffle
    all_data = domain_data + general_data
    random.shuffle(all_data)

    # Write output
    with open(out_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    # Stats
    label_counts: Dict[str, int] = {}
    for item in all_data:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1

    logger.info(f"Saved {len(all_data)} samples to {out_path}")
    logger.info(f"Label distribution: {json.dumps(label_counts, indent=2)}")
    logger.info(
        f"Domain: {len(domain_data)}, General: {len(general_data)}, "
        f"Ratio: {len(domain_data) / len(all_data):.1%}:"
        f"{len(general_data) / len(all_data):.1%}"
    )


if __name__ == "__main__":
    main()
