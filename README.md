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
- **Development Environment**: The local development environment uses Python 3.13.
- `faiss-cpu`: Bumped from `1.8.0` to `1.9.0.post1` due to unavailability on Python 3.13.
- `torch` & `transformers`: Bumped to `>=2.6.0` and `>=4.40.2` respectively due to Python 3.13 dropping support for older `torch` versions.
- `spacy`: Bumped from `3.7.4` to `>=3.8.0` due to Cython/GIL compatibility issues on Python 3.13.
- `numpy`, `streamlit`, `mlflow`, `duckdb`, & `pydantic`: Bumped to `>=` versions due to C-extension build issues and transitive dependency conflicts on Python 3.13.
