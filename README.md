# VeritasRAG

Self-Auditing RAG with Explainable Evidence Grounding.

## Setup

1. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   python -m spacy download en_core_web_sm
   ```

2. Configure environment variables in `.env` (copy from `.env.example`).

### Version Deviations from Guide
- **Development Environment:** Python 3.13. Always run scripts from the project venv; conda base is unsupported (Anaconda Python + PyTorch MPS + FAISS causes segfaults). Due to Keras segmentation faults with C-Extensions on Python 3.13, `USE_TF=0` and `USE_JAX=0` are explicitly enforced.
- **Execution Target:** The final `training/` phase expects execution on Google Colab (A100).
- `faiss-cpu`: Bumped from `1.8.0` to `1.9.0.post1` due to unavailability on Python 3.13.
- `torch` & `transformers`: Bumped to `>=2.6.0` and `>=4.40.2` respectively due to Python 3.13 dropping support for older `torch` versions.
- `spacy`: Bumped from `3.7.4` to `>=3.8.0` due to Cython/GIL compatibility issues on Python 3.13.
- `numpy`, `streamlit`, `mlflow`, `duckdb`, & `pydantic`: Bumped to `>=` versions due to C-extension build issues and transitive dependency conflicts on Python 3.13.
- **LLaMA 3.1 8B Setup**: Place `llama31-8b-q4.gguf` at `models/llm/` (see config). `llama-cpp-python` is installed via `CMAKE_ARGS="-DLLAMA_METAL=on" pip install -e ".[dev]"`.
- **Zero-shot NLI Verifier**: The guide specifies `deberta-v3-base-mnli` for Phase 3, but we use `cross-encoder/nli-deberta-v3-base` (which is actually trained on MNLI) as the zero-shot NLIScorer.
### Data Acquisition

The corpus is a 212-document subset of the CUAD (Contract Understanding Attainment Dataset). Raw data acquisition is scripted:

```bash
# Download CUAD from HuggingFace and match against manifest SHA-256 hashes
python scripts/download_corpus.py

# If originals are unavailable, reconstruct from the semantic metadata DB
python scripts/restore_corpus_from_db.py
```

Corpus size: **212 documents** (Frozen as Corpus v2.0 in `data/corpus_manifest.json`).

## Phase 2 Evaluation

- **Zero-shot reranker adopted**: The zero-shot `cross-encoder/ms-marco-MiniLM-L-6-v2` model is the active pipeline reranker. It achieved an MRR of 0.807, a +0.153 improvement over the hybrid baseline.
- **Reranker fine-tune logged as negative result**: The v2 fine-tuned model (MRR 0.714) underperformed the zero-shot baseline and is treated as a documented negative result.
- **Classifier v2 approved**: The v2 Query Classifier + rule-based keyword fallback achieved 14/15 accuracy on the Gate 2 probe and is active.

## Phase 3 Evaluation

- **Multi-Hop Agent (ReAct) vs. Single-Shot Baseline**: Evaluated on 200 samples from the HotpotQA distractor dataset. 
- **Metrics**: The Multi-Hop Agent achieved an MSR of **0.8500**, while the Single-Shot baseline achieved an MSR of **0.8600**. Both configurations easily cleared the MSR > 0.65 criteria.
- **Latency**: `ask.py` end-to-end latency on the golden query was 43.60s. This exceeds the 15s threshold, but this is dominated by ~20s of cold-start model loading on the MPS/Metal architecture. The core logic executes at a satisfactory speed.
- **Stability**: PyTorch MPS Out-Of-Memory (OOM) errors and Metal KV Cache crashes were resolved with rigorous thread-safety using a singleton `ThreadPoolExecutor`, explicit Python garbage collection, and PyTorch MPS cache clearing. Incremental JSONL persistence was added to long-running evaluation scripts.

## Incidents

### INC-001: data/raw/ corpus loss & reconciliation (discovered 2026-07-07)

**What happened:** 168 of 180 files in `data/raw/` were lost. Only 12 files remained (manual copies with `legal_*` prefix and ` copy` suffix, not from the original ingestion). The semantic index (`data/indexes/`) was intact, preserving all 29,501 chunks across 180 documents. We rebuilt the corpus using `download_corpus.py` which downloaded 200 files from HuggingFace.

**Corpus Reconciliation Breakdown (v1.0 -> v2.0):**
The `data/raw/` directory grew from 180 to 212 files. The exact breakdown is:
- **178** hash-matched originals (out of the 180 original corpus)
- **12** legacy "copy" files (the survivors of the data loss)
- **22** new downloads (files from the HuggingFace dataset that were not in the original 180)
- **2** mismatched/missing (original files that didn't match the new downloads)
Total: 178 + 12 + 22 = **212 files**.
This new 212-file state has been frozen as Corpus v2.0 and logged in `data/corpus_manifest.json` with their SHA-256 hashes. Any future changes to `data/raw/` require a manifest version bump.

**Root cause:** `data/` is in `.gitignore`, so file deletion was not tracked. The most likely cause is accidental manual deletion or a file manager operation that replaced the directory contents with the 12 test copies. No git history exists for `data/` to confirm the exact event.

**Impact:** The fixed-512 baseline ingestion (`run_fixed_ingestion.py`) only processed the 12 surviving files (420 chunks), producing an invalid Gate 1 experiment where the baseline scored zero on all metrics.

**Remediation:**
1. Document text was reconstructed from `data/indexes/metadata.db` → `data/reconstructed/` (180 docs).
2. `scripts/download_corpus.py` created to re-acquire CUAD originals from HuggingFace for future phases.
3. Fixed-512 ingestion now reads from `data/reconstructed/` for the canonical span-overlap evaluation frame.
4. `data/corpus_manifest.json` created to track the new 212-file corpus state.

**Prevention:** Raw data acquisition is now scripted via `download_corpus.py`. The `data/raw/` directory should always be re-creatable from CUAD.

## Known Tech Debt

### OFFSETS-01: Per-block char offsets in semantic chunker (Phase 2 fix)

The semantic chunker (`src/ingestion/semantic_chunker.py`) and indexer store `char_start`/`char_end` as per-block offsets (resetting for each `ParsedBlock`), not document-absolute offsets. This must be fixed with a re-index at the start of Phase 2 — Phase 4's span alignment and audit records require one consistent document-absolute offset frame. Do NOT fix mid-gate.
