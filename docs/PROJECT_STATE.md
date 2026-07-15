# VeritasRAG — Project State (Handoff Document)

_Last updated: 2026-07-12. Read this file + the repo to resume with zero chat history._

---

## 1. Phase / Gate Status

| Phase | Gate | Status | Key Verified Metrics |
|-------|------|--------|---------------------|
| **1 — Ingestion & Retrieval** | Gate 1 | **CLOSED ✅** | Semantic MRR 0.6535, Recall@5 0.7700; Fixed-512 baseline MRR 0.5752. 100/100 golden remap. Corpus v2.0 frozen (212 docs). |
| **2 — Reranker & Classifier** | Gate 2 | **CLOSED ✅** | Zero-shot reranker MRR **0.8068**; fine-tuned v2 MRR 0.7141 (negative result, documented). Classifier v2 14/15 with keyword fallback. |
| **3 — Generation & Multi-Hop Agent** | Gate 3 | **CLOSED ✅** | Agent 0.5909 vs Single-shot 0.4909 on hard subset (+0.10). Overall 0.85 vs 0.86. 4 catastrophic failures (parallel decomposition limitation). |
| **4 — Evaluation, Explainability, Monitoring** | Gate 4 | **NOT STARTED** | `src/verification/`, `src/audit/`, `src/monitoring/` are empty `__init__.py` stubs. Phase 4 plan has not been created. |

Results files: `evaluation/benchmarks/results_phase{1,2,2_v2,3}.json`.

---

## 2. Active Work / Next Concrete Action

**Gate 3 is CLOSED.** Phase 4 starts next.
 
 1. **Phase 4 Implementation Plan:** Create the detailed plan for the claim-level NLI verifier, span alignment, selective regeneration, audit record, and query_pipeline.
 2. **Phase 2 Cleanup Ticket:** (§3) should be cleared alongside Phase 4 work.

---

## 3. Pending Tickets

### Phase 2 Cleanup Ticket (deferred, still open)
- [ ] **mlflow → sqlite backend**: `training/train_reranker.py` and `train_query_classifier.py` both already use `sqlite:///mlflow.db`. Pin `mlflow` version exactly in the lock file.
- [ ] **`report_to="none"`**: `train_query_classifier.py` already has it. `train_reranker.py` uses `CrossEncoder` (no native `report_to`); add a comment or the WANDB_DISABLED env note.
- [ ] **CPU-only 3-sample dry-run test** per training script (catches API breakage like `eval_strategy` before Colab).
- [ ] **Unify training CLIs on Hydra**: `generate_training_data.py` still uses `argparse`; the others use Hydra or custom args. Standardize.

### OFFSETS-01 (DONE)
 Per-block `char_start`/`char_end` in the semantic chunker have been updated to be document-absolute. Re-indexing is complete and verified. Golden set mappings have been fully updated.

---

## 4. Confirmed Decisions (Still in Force)

| Decision | Detail |
|----------|--------|
| **Reranker** | Zero-shot `cross-encoder/ms-marco-MiniLM-L-6-v2` (config `reranker.mode: zeroshot`). Fine-tuned is a documented negative result. |
| **NLI Verifier** | `cross-encoder/nli-deberta-v3-base` (guide correction recorded in README). |
| **LLM** | Llama 3.1 8B Q4 GGUF at `models/llm/`, via `llama-cpp-python` with Metal. `n_ctx: 8192` from config; `context_assembler` derives token budget from it. |
| **LLM Threading** | Singleton `ThreadPoolExecutor(max_workers=1)` in `model_registry.py`. All `llama.cpp` calls serialized on one OS thread. No `asyncio.wait_for` — timeout is a post-hoc watchdog. |
| **Corpus** | CUAD v2.0 — 212 docs, frozen in `data/corpus_manifest.json` (SHA-256 per file). |
| **Golden Set** | `evaluation/benchmarks/golden_set.json` — 100 queries, HUMAN_VERIFIED. **READ-ONLY.** |
| **No Frameworks** | No LangChain / LlamaIndex / Haystack. All core algorithms from scratch (Rule 10). |
| **Device** | MPS (Metal) if available, else CPU. Never CUDA. `faiss-cpu` only. |

---

## 5. Environment Gotchas

- **venv, not conda.** Anaconda Python + PyTorch MPS + FAISS segfaults. Always `source .venv/bin/activate`.
- **Python 3.13** — requires `USE_TF=0 USE_JAX=0` to avoid Keras C-extension segfaults.
- **llama-cpp-python install**: `CMAKE_ARGS="-DLLAMA_METAL=on" pip install -e ".[dev]"`.
- **Metal concurrency**: never use `asyncio.Lock` or `asyncio.to_thread` for llama calls. The `ThreadPoolExecutor` pattern in `model_registry.py` is the fix. Cancelling a C-thread via `asyncio.wait_for` creates zombie decodes → `llama_decode -3` crash.
- **MPS OOM in long evals**: call `gc.collect()` + `torch.mps.empty_cache()` per iteration. See `run_gate3_eval.py`.
- **Process-exit Metal assert** (`GGML_ASSERT rsets->data count == 0`): cosmetic crash at interpreter shutdown; does not affect results. Ignore it.
- **Colab training**: scripts in `training/` are standalone (no local imports). Version-pin `sentence-transformers`, `transformers`, `datasets` in Colab cells to match local.
- **MLflow**: if mlflow complains about file stores, set `MLFLOW_ALLOW_FILE_STORE=true` or use `sqlite:///mlflow.db`.

---

## 6. Authoritative File Pointers

| What | Path |
|------|------|
| Build guide (single source of truth) | `ANTIGRAVITY_BUILD_GUIDE.md` (root) |
| Project rules | `.agents/AGENTS.md` |
| Runtime config | `configs/config.yaml` |
| Corpus manifest | `data/corpus_manifest.json` |
| Golden set (READ-ONLY) | `evaluation/benchmarks/golden_set.json` |
| Gate 1 results | `evaluation/benchmarks/results_phase1.json` |
| Gate 2 results (v2, final) | `evaluation/benchmarks/results_phase2_v2.json` |
| Gate 3 results | `evaluation/benchmarks/results_phase3.json` + `.jsonl` |
| Training scripts | `training/{train_reranker,train_query_classifier,generate_training_data}.py` |
| Model weights | `models/{llm,reranker,reranker_v2,query_classifier,query_classifier_v2,nli_verifier}/` |
| Incident log | `README.md` § Incidents (INC-001) |
