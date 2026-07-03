RULES FOR THIS PROJECT — NON-NEGOTIABLE

1. Build ONE phase at a time, ONE file at a time, in the exact order listed in
   Section 5 of ANTIGRAVITY_BUILD_GUIDE.md. Never skip ahead.
2. Every file must be complete and runnable — no pseudocode, no `pass` placeholders.
   Deferred work: `# TODO (Phase N): reason`.
3. Code standards: PEP 8, PEP 257 docstrings, type hints on every signature,
   Pydantic v2 for schemas, loguru for logging (never print()), pathlib.Path for paths.
4. ZERO hardcoded tunables in src/ — every threshold, top-k, model name, path
   comes from configs/config.yaml via Hydra/OmegaConf.
5. Device: this is an M5 Mac. PyTorch device = "mps" if available else "cpu".
   NEVER cuda. FAISS = faiss-cpu. llama.cpp built with Metal.
6. tests/ mirrors src/. After finishing each file, write its test file and run
   `pytest tests/test_<name>.py`. A phase is DONE only when all its tests pass.
7. After each file, run `ruff check` and `black --check` and fix violations.
8. training/ scripts must run standalone on Colab A100 — no imports from local
   state, all inputs via CLI args + configs/training_config.yaml.
9. Secrets ONLY in .env (loaded via python-dotenv). Never in YAML configs, never
   committed. deployment_config.yaml holds ports/paths only.
10. Do not add frameworks (LangChain, LlamaIndex, Haystack). Every core algorithm
    (chunker, RRF, CUSUM, span alignment) is implemented from scratch — that is
    the point of this project.
11. Commit after every completed file: `feat(phaseN): <file> — <what it does>`.
