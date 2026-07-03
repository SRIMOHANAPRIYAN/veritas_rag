# VeritasRAG — Complete Project Context
### Self-Auditing RAG with Explainable Evidence Grounding
> **For use inside Claude Projects. Feed this file as project knowledge so Claude has full context at every session.**

---

## 0. Who I Am & How I Work

| Attribute | Detail |
|---|---|
| **Role** | Generative AI Engineer (Internship) |
| **Level** | Intermediate — comfortable with APIs and fine-tuning; learning model internals |
| **Goal** | Land a role at a top product-based company (Google, Meta, Amazon) |
| **Device** | MacBook Pro M5 · 16 GB unified memory · 512 GB SSD · 10-core CPU · 10-core GPU |
| **IDE** | VS Code (local) for all code · Google Colab A100 only for GPU training runs |
| **Python** | 3.11+ · virtual environment via `venv` · installed with `pip install -e ".[dev]"` |

### How Claude Must Work With Me
- **Phase-by-phase, file-by-file.** Never dump all code at once. One file at a time, in phase order.
- **File path at the top of every code block.** Example: `src/ingestion/document_parser.py`
- **Explain every piece of code.** After each code block, explain what it does, why it's written that way, and any design trade-offs.
- **Tech stack comparisons.** Whenever introducing a library or model, compare it to alternatives and explain why this specific choice was made for VeritasRAG.
- **Industry-standard code.** Clean, readable, PEP 8, typed, documented with docstrings. No shortcuts.
- **DSA where applicable.** If a data structure or algorithm improves performance, use it and explain the complexity gain.
- **M5 Mac awareness.** The M5 chip has a unified memory architecture — CPU and GPU share the same memory pool. This means `mps` (Metal Performance Shaders) can be used as a PyTorch device for local inference. Never recommend CUDA for local runs; use `mps` or `cpu`.

---

## 1. Why This Project Exists — The Problem

### The Core Finding
**73% of all RAG failures happen at retrieval, not generation.**

Production RAG systems fail silently. They retrieve wrong documents and then generate confident, well-structured answers grounded in those wrong documents. Standard monitoring dashboards (latency, error rate, throughput) show nothing wrong.

### Five Documented Failure Modes

| # | Failure | Evidence |
|---|---|---|
| 1 | **Wrong documents retrieved confidently** | 40–60% of RAG projects fail to reach production due to retrieval quality issues |
| 2 | **No claim-level grounding** | Existing systems show `(Source: doc.pdf)` — not which sentence supports which claim |
| 3 | **Silent retrieval drift** | Embedding quality degrades as corpus changes; no production tool monitors this |
| 4 | **Multi-hop query collapse** | Single-shot retrieval fails entirely on queries requiring cross-document reasoning |
| 5 | **No auditability for regulated industries** | Legal, medical, financial deployments blocked by inability to explain answers to auditors |

### Gap in Existing Tools

| Tool | Solves | Misses |
|---|---|---|
| LangChain / LlamaIndex | Pipeline scaffolding | No drift monitor, no claim-level grounding |
| RAGAS / DeepEval | Batch evaluation metrics | Offline only — not real-time, not claim-level |
| GraphRAG (Microsoft) | Graph-based relational retrieval | No faithfulness verifier, no audit trail |
| RT4CHART (arXiv Mar 2026) | Claim decomposition + NLI | Research prototype only — not deployable |

---

## 2. Our Solution — VeritasRAG

### One-Line Description
> A self-auditing RAG system that treats retrieval as a traceability problem — every claim in every generated answer must be provably anchored to a specific sentence in the retrieved context.

### Four Core Design Principles
1. **Retrieval as Traceability** — every chunk carries metadata: source doc, page, section path, chunk ID. Nothing is anonymous.
2. **Claim-Level Honesty** — faithfulness is measured per atomic claim, not per answer. Contradicted claims are flagged and selectively regenerated.
3. **Self-Monitoring** — the system continuously evaluates its own retrieval quality and fires alerts before users notice degradation.
4. **Audit-First Design** — every query produces a machine-readable JSON audit record queryable by engineers and compliance teams.

### What Makes VeritasRAG Unique (Novelty Proof)

| Feature | VeritasRAG | LangChain | GraphRAG | RAGAS | RT4CHART |
|---|---|---|---|---|---|
| Adaptive semantic chunker | ✅ from scratch | Partial | ✗ | ✗ | ✗ |
| RRF hybrid retrieval | ✅ from scratch | Plugin | ✗ | ✗ | ✗ |
| Fine-tuned cross-encoder reranker | ✅ trained | ✗ | ✗ | ✗ | ✗ |
| Multi-hop iterative retrieval | ✅ from scratch | Agent chains | Partial | ✗ | ✗ |
| Claim-level NLI verification | ✅ fine-tuned | ✗ | ✗ | Answer-level only | Research only |
| Contradiction-triggered selective regen | ✅ novel | ✗ | ✗ | ✗ | ✗ |
| CUSUM retrieval drift monitor | ✅ novel | ✗ | ✗ | ✗ | ✗ |
| Structured audit trail (regulatory-grade) | ✅ novel | ✗ | ✗ | ✗ | ✗ |

### Three Genuinely Novel Contributions
1. **CUSUM-Based Retrieval Drift Detection** — First application of Statistical Process Control to RAG retrieval quality monitoring. No prior published system does this.
2. **Contradiction-Triggered Selective Regeneration** — Only the contradicted sentence is regenerated, not the whole answer. Novel efficiency-correctness trade-off.
3. **Corpus Delta Attribution** — When drift is detected, ranks suspect documents by cosine distance to regressions in the golden set. Directly actionable for corpus maintainers.

---

## 3. Project Structure

```
veritas_rag/                            ← project root
│
├── src/                                ← ALL source code lives here
│   ├── ingestion/                      ← offline: parse → chunk → index
│   │   ├── __init__.py
│   │   ├── document_parser.py          ← PDF / DOCX / HTML / TXT parsers
│   │   ├── semantic_chunker.py         ← cosine-boundary chunker
│   │   ├── metadata_tagger.py          ← chunk metadata schema (Pydantic)
│   │   └── indexer.py                  ← FAISS + BM25 + SQLite builder
│   │
│   ├── retrieval/                      ← online: query → fetch → rerank
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py         ← FAISS + BM25 + RRF fusion
│   │   ├── reranker.py                 ← cross-encoder reranker (Phase 2)
│   │   └── query_classifier.py         ← simple vs multi-hop classifier
│   │
│   ├── verification/                   ← claim decompose → NLI verify → span align
│   │   ├── __init__.py
│   │   ├── claim_decomposer.py         ← spaCy dep-parse → atomic claims
│   │   ├── nli_verifier.py             ← fine-tuned DeBERTa NLI (Phase 4)
│   │   ├── span_aligner.py             ← char-level evidence pointer
│   │   └── selective_regen.py          ← contradiction-triggered regen
│   │
│   ├── generation/                     ← LLM wrapper + prompt templates
│   │   ├── __init__.py
│   │   ├── llm_client.py               ← llama.cpp Python bindings
│   │   ├── prompt_templates.py         ← all system/user prompt strings
│   │   └── context_assembler.py        ← formats chunks into LLM context
│   │
│   ├── agents/                         ← multi-hop iterative retrieval
│   │   ├── __init__.py
│   │   └── iterative_retrieval_agent.py
│   │
│   ├── monitoring/                     ← drift detection (Phase 5)
│   │   ├── __init__.py
│   │   ├── cusum_monitor.py            ← CUSUM drift detection
│   │   ├── golden_set_manager.py       ← golden eval set I/O
│   │   └── corpus_delta.py             ← drift attribution to documents
│   │
│   ├── audit/                          ← audit trail schema + storage
│   │   ├── __init__.py
│   │   ├── audit_record.py             ← Pydantic AuditRecord schema
│   │   ├── audit_writer.py             ← async DuckDB writer
│   │   └── audit_query.py              ← analytics query helpers
│   │
│   └── pipeline/                       ← orchestrates all stages end-to-end
│       ├── __init__.py
│       ├── ingestion_pipeline.py        ← parse → chunk → index (batch)
│       └── query_pipeline.py            ← query → retrieve → verify → answer
│
├── training/                           ← fine-tuning scripts (Colab A100 only)
│   ├── train_reranker.py               ← cross-encoder fine-tuning (Phase 2)
│   ├── train_nli_verifier.py           ← DeBERTa NLI fine-tuning (Phase 4)
│   ├── train_query_classifier.py
│   └── generate_training_data.py       ← hard negatives + domain NLI pairs
│
├── tests/                              ← mirrors src/ exactly
│   ├── __init__.py
│   ├── test_document_parser.py
│   ├── test_semantic_chunker.py
│   ├── test_indexer.py
│   ├── test_hybrid_retriever.py
│   └── test_metrics.py
│
├── evaluation/                         ← benchmarking (separate from src)
│   ├── metrics.py                      ← precision@k, recall@k, MRR, CCR, CR
│   ├── eval_runner.py
│   ├── benchmark_comparison.py
│   └── benchmarks/
│       ├── golden_set.json             ← 100 queries + correct chunk IDs
│       └── veritasqa_test_set.json     ← 200 claims with manual verdicts
│
├── api/                                ← FastAPI backend (Phase 5)
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── ui/                                 ← Streamlit dashboard (Phase 5)
│   └── app.py
│
├── configs/
│   ├── config.yaml                     ← runtime hyperparameters
│   ├── training_config.yaml            ← training hyperparameters
│   └── deployment_config.yaml          ← ports, model paths, API keys
│
├── data/                               ← gitignored
│   ├── raw/                            ← source documents
│   ├── processed/                      ← parsed ParsedBlock JSONs
│   ├── chunks/                         ← Chunk JSONs after chunking
│   └── indexes/                        ← faiss.index, bm25.pkl, metadata.db
│
├── models/                             ← gitignored
│   ├── reranker/                       ← fine-tuned cross-encoder (Phase 2)
│   ├── nli_verifier/                   ← fine-tuned DeBERTa (Phase 4)
│   ├── query_classifier/
│   └── llm/                            ← llama31-8b-q4.gguf (~4.7GB)
│
├── notebooks/                          ← exploration only, never imported
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunker_calibration.ipynb
│   └── 03_retrieval_analysis.ipynb
│
├── scripts/                            ← thin entry-point wrappers
│   ├── run_ingestion.py
│   ├── run_evaluation.py
│   ├── run_drift_monitor.py
│   └── calibrate_threshold.py
│
├── .github/workflows/drift_monitor.yml ← GitHub Action: daily drift check
├── .env.example
├── .gitignore
├── pyproject.toml                      ← replaces requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Key Structural Rules
1. **`src/` layout** — all importable code lives under `src/`. Install with `pip install -e .` so imports work everywhere without `sys.path` hacks.
2. **Separate by concern, not by phase** — folders are `ingestion/`, `retrieval/`, `verification/` etc. New phases add files to existing folders.
3. **`training/` is isolated** — not in `src/` because training scripts run once on Colab, not in production.
4. **Zero hardcoded values in `src/`** — every threshold, top-k, model name lives in `configs/config.yaml`.
5. **`data/` and `models/` are gitignored** — always, no exceptions.
6. **`tests/` mirrors `src/`** — `src/ingestion/semantic_chunker.py` → `tests/test_semantic_chunker.py`.
7. **Notebooks never contain production logic** — useful code moves to `src/` immediately.

---

## 4. Tech Stack

### Core ML / NLP

| Component | Choice | Why |
|---|---|---|
| Sentence Embedding | `sentence-transformers` · `all-mpnet-base-v2` | Best quality/speed on MTEB for retrieval; 768-dim; runs locally free |
| Dense Index | `faiss-cpu` · `IndexFlatIP` | Exact search; fast enough for <500K chunks; swap to HNSW only if needed |
| Sparse Retrieval | `rank-bm25` · BM25Okapi | Catches exact matches (product codes, proper nouns) that dense embeddings miss |
| Cross-Encoder Reranker | `ms-marco-MiniLM-L-6-v2` (fine-tuned) | Small but powerful; fine-tuning on domain gives 8–12 F1 gain over zero-shot |
| NLI Verifier | `DeBERTa-v3-base` (fine-tuned on MNLI) | Best-in-class NLI accuracy; runs on CPU for inference |
| LLM Generator | `Llama 3.1 8B Instruct` · GGUF Q4_K_M via `llama-cpp-python` | Open-source; runs locally on M5 Mac via `mps`; no API cost |
| NLP Preprocessing | `spaCy` · `en_core_web_sm` | Dependency parsing for atomic claim decomposition |

### Local Device Notes (M5 Mac)
- **Use `device = "mps"`** for PyTorch inference (sentence-transformers, DeBERTa). The M5 GPU shares unified memory — no VRAM limit separate from RAM.
- **FAISS runs on CPU** — `faiss-cpu` is correct. There's no official Apple Silicon FAISS GPU build.
- **llama.cpp** uses Metal (M5 GPU) automatically when installed with `CMAKE_ARGS="-DLLAMA_METAL=on"`.
- **Training** still goes to Colab A100 — M5 GPU is excellent for inference but too slow for fine-tuning full models.

### Storage

| Component | Choice | Why |
|---|---|---|
| Audit trail | DuckDB | Embedded columnar DB; fast analytics; no server needed |
| Chunk metadata | SQLite | Lightweight KV; built into Python stdlib |
| Experiment tracking | MLflow | Local; track fine-tuning runs, metrics, model versions |
| Config | Hydra / OmegaConf | Composable config files; essential for reproducibility |

### API & Deployment

| Component | Choice |
|---|---|
| Backend | FastAPI 0.111+ |
| Frontend | Streamlit 1.35+ |
| Containerization | Docker + Docker Compose |
| Cloud hosting | Hugging Face Spaces (free GPU) |
| Drift monitor CI | GitHub Actions (daily cron) |

### Dependencies (`pyproject.toml`)
```toml
[project]
name = "veritas-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sentence-transformers==2.7.0",
    "faiss-cpu==1.8.0",
    "rank-bm25==0.2.2",
    "torch==2.3.0",
    "transformers==4.40.2",
    "spacy==3.7.4",
    "pdfplumber==0.11.0",
    "python-docx==1.1.2",
    "beautifulsoup4==4.12.3",
    "duckdb==0.10.3",
    "pydantic==2.7.1",
    "pyyaml==6.0.1",
    "numpy==1.26.4",
    "tqdm==4.66.4",
    "loguru==0.7.2",
    "fastapi==0.111.0",
    "uvicorn==0.29.0",
    "streamlit==1.35.0",
    "hydra-core==1.3.2",
    "mlflow==2.12.2",
    "llama-cpp-python",   # install separately with Metal flag
]

[project.optional-dependencies]
dev = ["pytest==8.2.0", "pytest-cov==5.0.0", "black==24.4.2", "ruff==0.4.4", "ipykernel"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

---

## 5. System Architecture

### Two Pipelines + One Background Job

```
INGESTION PIPELINE (Offline)          QUERY PIPELINE (Online, ~300–800ms)
───────────────────────────           ──────────────────────────────────────
Raw Documents                         User Query
  → Document Parser                     → Query Classifier (simple/multi-hop)
  → Semantic Chunker                    → Hybrid Retriever (FAISS + BM25 + RRF)
  → Metadata Tagger                     → Cross-Encoder Reranker
  → Embedding Model                     [if multi-hop → Iterative Agent]
  → FAISS Index                         → Context Assembler
  → BM25 Index                          → LLM Generator (Llama 3.1 8B)
  → SQLite Metadata Store               → Claim Decomposer (spaCy)
                                         → NLI Verifier (DeBERTa-v3)
                                         [if contradicted → Selective Regen]
DRIFT MONITOR (Daily GitHub Action)     → Answer + Audit Trail JSON
───────────────────────────
Golden Set Retrieval
  → CUSUM Analysis
  → Drift Alert Logger
  → Corpus Delta Attribution
  → Remediation Suggestion
```

### Data Flow — 9 Steps

1. **Document Parsing** — format-specific parsers extract `(text, metadata)` tuples preserving heading hierarchy, page numbers, table flags.
2. **Semantic Chunking** — cosine similarity boundary detection between consecutive sentence embeddings. Adaptive threshold calibrated per corpus.
3. **Dual Indexing** — chunks embedded with `all-mpnet-base-v2` → FAISS + simultaneously tokenized → BM25. Both persisted to disk.
4. **Hybrid Retrieval** — FAISS top-100 + BM25 top-100 fused via Reciprocal Rank Fusion (RRF, k=60) → top-50 candidates.
5. **Cross-Encoder Reranking** — fine-tuned cross-encoder scores query-chunk pairs jointly → top-8 with uncertainty scores.
6. **Multi-Hop Expansion (conditional)** — if query classifier triggers: LLM decomposes query → sub-questions → per-sub-question retrieval → NLI coverage termination (max 3 iterations).
7. **LLM Generation** — Llama 3.1 8B with `[CITE:chunk_id]` inline citation prompt → answer text.
8. **Claim Decomposition + NLI Verification** — spaCy extracts atomic claims → DeBERTa NLI verifies each claim → verdict: `ENTAILED / CONTRADICTED / BASELESS` + evidence span pointer.
9. **Selective Regeneration + Audit Write** — contradicted claims regenerated in isolation → full `AuditRecord` written to DuckDB.

---

## 6. Complete Implementation Roadmap

### Phase 1 — Semantic Chunker + Hybrid Retriever (Weeks 1–3)
**Deliverable:** Retriever that beats naive chunking on benchmark dataset

Files to build (in order):
- `src/ingestion/document_parser.py`
- `src/ingestion/semantic_chunker.py`
- `src/ingestion/metadata_tagger.py`
- `src/ingestion/indexer.py`
- `src/retrieval/hybrid_retriever.py`
- `evaluation/metrics.py`
- `scripts/run_ingestion.py`
- `scripts/calibrate_threshold.py`
- `tests/test_document_parser.py`
- `tests/test_semantic_chunker.py`
- `tests/test_hybrid_retriever.py`

Key milestone: precision@5 and recall@5 benchmark showing semantic chunking outperforms 512-token fixed chunking. Build 100-query golden set.

**Tech:** spaCy, sentence-transformers, FAISS, rank-bm25, NumPy, SQLite

### Phase 2 — Cross-Encoder Reranker Fine-tuning (Weeks 4–6)
**Deliverable:** Fine-tuned reranker with 8+ MRR gain over zero-shot baseline

Files to build (in order):
- `training/generate_training_data.py`
- `training/train_reranker.py`
- `training/train_query_classifier.py`
- `src/retrieval/reranker.py`
- `src/retrieval/query_classifier.py`
- `tests/test_indexer.py`

Run training on Colab A100 (~2 hours). Download weights to `models/reranker/`.

**Tech:** HuggingFace Transformers, CrossEncoder, MS MARCO dataset, MLflow

### Phase 3 — Multi-Hop Iterative Retrieval Agent (Weeks 7–8)
**Deliverable:** System handles multi-hop queries where single-shot RAG fails

Files to build:
- `src/agents/iterative_retrieval_agent.py`
- `src/generation/llm_client.py`
- `src/generation/prompt_templates.py`
- `src/generation/context_assembler.py`
- `src/pipeline/ingestion_pipeline.py`

Evaluate on HotpotQA. Target: multi-hop recall@5 > 0.65 vs baseline 0.42.

**Tech:** llama-cpp-python (Metal), llama.cpp, HotpotQA dataset, ReAct loop

### Phase 4 — Claim-Level NLI Verifier (Weeks 9–11)
**Deliverable:** Verifier assigns ENTAILED/CONTRADICTED/BASELESS per claim with evidence span

Files to build:
- `training/train_nli_verifier.py`
- `src/verification/claim_decomposer.py`
- `src/verification/nli_verifier.py`
- `src/verification/span_aligner.py`
- `src/verification/selective_regen.py`
- `src/pipeline/query_pipeline.py`
- `src/audit/audit_record.py`

Run DeBERTa fine-tuning on Colab A100 (~4 hours). Build VeritasQA test set (200 manually labeled claims). Target: >85% verdict agreement with human labels.

**Tech:** DeBERTa-v3-base, MultiNLI + SNLI, domain synthetic NLI pairs, character-level span alignment

### Phase 5 — Drift Monitor + Audit Trail + Deployment (Weeks 12–14)
**Deliverable:** Deployed system with live audit dashboard and ≥5 real users

Files to build:
- `src/monitoring/cusum_monitor.py`
- `src/monitoring/golden_set_manager.py`
- `src/monitoring/corpus_delta.py`
- `src/audit/audit_writer.py`
- `src/audit/audit_query.py`
- `api/main.py`
- `api/schemas.py`
- `ui/app.py`
- `.github/workflows/drift_monitor.yml`
- `Dockerfile`
- `docker-compose.yml`

Deploy to HuggingFace Spaces. Onboard real users. Write 4-page technical report.

**Tech:** CUSUM (NumPy from scratch), DuckDB, FastAPI, Streamlit, Docker, GitHub Actions, HuggingFace Spaces

---

## 7. Core Algorithms — Technical Reference

### Semantic Chunking Algorithm
```
Input: document text
1. Segment into sentences (spaCy)
2. Encode all sentences in batch (all-mpnet-base-v2, normalized)
3. Compute cosine similarity between consecutive sentence pairs
4. Boundary = similarity < threshold AND current_tokens >= min_tokens
5. Hard boundary at max_tokens regardless of similarity
6. Tables: always pass through as a single chunk, never split
Output: List[Chunk] with full metadata
```

### Reciprocal Rank Fusion
```
score(chunk) = Σ  1 / (k + rank_i)   for each list containing chunk
k = 60   ← standard constant; smooths dominance of top-ranked items
```

### CUSUM Drift Monitor
```
S_t = max(0, S_{t-1} + (baseline_precision - daily_precision - slack))
slack = 0.02   ← sensitivity tuner
alert_threshold = 5.0
Alert fires when S_t > threshold
```

### NLI Verdict Assignment
```
For each atomic claim vs each retrieved chunk:
  - entailment_score > 0.80  → ENTAILED (record evidence span)
  - contradiction_score > 0.70 AND best_entailment < 0.20  → CONTRADICTED
  - else  → BASELESS
Contradicted claims → selective regeneration (only that sentence)
```

### DSA Applications in VeritasRAG
| Location | DSA Used | Why |
|---|---|---|
| RRF fusion | Hash map (dict) for score accumulation | O(n) merge vs O(n log n) sort-then-merge |
| Chunk ID map | Array (list) with direct index access | FAISS returns integer indices → O(1) lookup |
| CUSUM | Sliding statistic (running max) | O(1) per update, no history storage needed |
| BM25 retrieval | Inverted index (internal to rank-bm25) | O(|query_terms| × avg_postings) vs O(n) linear scan |
| Span alignment | Two-pointer / character cursor | O(n) span finding vs O(n²) brute force |
| Golden set lookup | Set for O(1) relevant chunk ID checks | Used in precision@k and recall@k metrics |

---

## 8. Evaluation Framework

### Four Custom Metrics (Not in RAGAS or DeepEval)

| Metric | Formula | Target |
|---|---|---|
| **Claim Coverage Rate (CCR)** | `|ENTAILED claims| / |total claims|` | > 0.85 |
| **Contradiction Rate (CR)** | `|CONTRADICTED claims| / |total claims|` | < 0.04 (baseline naive RAG: ~0.18) |
| **Drift Alert Precision (DAP)** | `True Drift Alerts / Total Drift Alerts` | > 0.80 |
| **Multi-hop Sub-question Recall (MSR)** | `mean(recall@5 per sub-question)` on HotpotQA | > 0.65 (baseline single-shot: ~0.42) |

### Standard Retrieval Metrics
- `precision@k` — fraction of top-k retrieved that are relevant
- `recall@k` — fraction of all relevant in top-k
- `MRR (Mean Reciprocal Rank)` — 1/rank of first relevant result

### Datasets

| Dataset | Phase | Used For |
|---|---|---|
| MS MARCO | 2 | Cross-encoder fine-tuning |
| MultiNLI + SNLI | 4 | DeBERTa NLI base training |
| HotpotQA | 3 | Multi-hop retrieval evaluation |
| 2WikiMultiHopQA | 3 | Harder multi-hop eval |
| CUAD (legal) / PubMedQA (medical) / FinQA (financial) | 1 | Domain corpus |
| Golden set (hand-built) | 1, 5 | Retrieval eval + drift monitor baseline |
| VeritasQA (hand-built) | 4 | NLI verifier evaluation |

---

## 9. Audit Trail Schema

```python
class ClaimVerdict(BaseModel):
    claim_text: str
    verdict: Literal["ENTAILED", "CONTRADICTED", "BASELESS"]
    evidence_chunk_id: str | None
    evidence_span_start: int | None   # char offset in chunk text
    evidence_span_end: int | None
    confidence: float

class AuditRecord(BaseModel):
    query_id: str
    timestamp: datetime
    query_text: str
    retrieval_candidates: List[RetrievalCandidate]  # all scored chunks
    chunks_used: List[str]            # chunk_ids used in generation
    answer_raw: str
    claims: List[ClaimVerdict]
    answer_final: str
    contradiction_regenerations: int
    pipeline_latency_ms: int
    drift_score_at_query: float
    multi_hop_triggered: bool
```

---

## 10. Interview Stories

### Google — Most Complex Component
> "The claim-level NLI verifier. I implemented a dependency-parsing-based decomposer using spaCy that extracts subject-verb-object triples as self-contained propositions. Each claim is then run through fine-tuned DeBERTa-v3 — not a prompted LLM, which would be circular — to get an Entailed/Contradicted/Baseless verdict. The hardest part was the character-level span alignment that maps each verdict back to the exact sentence in the retrieved chunk."

### Meta — Measurement
> "I built four evaluation metrics that don't exist in RAGAS or DeepEval: Claim Coverage Rate, Contradiction Rate, Drift Alert Precision, and Multi-hop Sub-question Recall. On HotpotQA, my system improved multi-hop recall from 0.42 to 0.67 over single-shot RAG. Contradiction Rate dropped from 18% to 4%."

### Amazon — Production Engineering
> "I applied CUSUM control charts from statistical process control to detect retrieval drift. The monitor runs daily against a held-out golden set. When S_t exceeds 5.0, a DriftAlert fires with corpus delta attribution — ranking which recently added documents most likely caused the drift. Zero query-time latency added."

### Any Company — Differentiation
> "Three ways this differs from LangChain + RAGAS: (1) I built every component from scratch — full control. (2) RAGAS evaluates offline in batch. VeritasRAG verifies per claim in real time and corrects contradictions before the user sees them. (3) RAGAS gives a score. My system gives a structured audit trail with a verdict, evidence pointer, and source chunk ID per claim — that's the difference between a metric and an explanation."

---

## 11. VS Code + Colab Split

| Task | Where | Reason |
|---|---|---|
| All Python code writing | Local VS Code | IDE, git, debugger, file system |
| Document parsing, chunking, indexing | Local VS Code | CPU-only, M5 handles easily |
| FAISS retrieval, BM25, RRF | Local VS Code | In-memory, no GPU needed |
| Evaluation harness | Local VS Code | Pure Python |
| **Cross-encoder fine-tuning (Phase 2)** | **Colab A100** | GPU needed, ~2 hrs |
| **DeBERTa NLI fine-tuning (Phase 4)** | **Colab A100** | GPU needed, ~4 hrs |
| LLM inference (Llama 3.1 8B) | Local VS Code | M5 GPU via Metal/MPS |
| FastAPI, Streamlit | Local VS Code | Runs on localhost |

### Colab Workflow
1. Write training scripts locally in VS Code
2. Push to GitHub
3. Open Colab → clone repo → run `training/*.py`
4. Save weights to Google Drive
5. Download weights locally to `models/`
6. Continue development in VS Code

---

## 12. configs/config.yaml Reference

```yaml
chunker:
  model: "all-mpnet-base-v2"
  similarity_threshold: 0.6       # calibrate per corpus: run scripts/calibrate_threshold.py
  min_tokens: 150
  max_tokens: 512
  batch_size: 64

indexer:
  embedding_model: "all-mpnet-base-v2"
  embedding_dim: 768
  faiss_index_path: "data/indexes/faiss.index"
  bm25_index_path: "data/indexes/bm25.pkl"
  metadata_db_path: "data/indexes/metadata.db"

retriever:
  top_k_dense: 100
  top_k_sparse: 100
  top_k_fused: 50
  rrf_k: 60

reranker:
  model_path: "models/reranker/"
  top_k_reranked: 8

generation:
  model_path: "models/llm/llama31-8b-q4.gguf"
  device: "mps"                   # M5 Mac GPU via Metal
  max_tokens: 512
  temperature: 0.0

verification:
  nli_model_path: "models/nli_verifier/"
  entailment_threshold: 0.80
  contradiction_threshold: 0.70
  device: "mps"

monitoring:
  golden_set_path: "evaluation/benchmarks/golden_set.json"
  cusum_slack: 0.02
  cusum_alert_threshold: 5.0

evaluation:
  metrics: ["precision@5", "recall@5", "mrr"]
```
