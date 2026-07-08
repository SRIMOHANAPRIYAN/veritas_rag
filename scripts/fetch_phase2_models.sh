#!/usr/bin/env bash
# Fetch Phase 2 trained models from Google Drive (file IDs resolved 2026-07-08).
# Skips checkpoint-*/eval/training folders — only the final model files.
# Usage: bash scripts/fetch_phase2_models.sh   (run from repo root, venv active)
set -euo pipefail

command -v gdown >/dev/null 2>&1 || pip install gdown

mkdir -p models/reranker models/query_classifier

echo "── Reranker (ms-marco-MiniLM fine-tuned, ~87 MB) ──"
gdown 1BvV5EG4nEX5vLzbdebaQxrHHG2i6QdZW -O models/reranker/config.json
gdown 1_55klgHK8g__ZDyxEvrPfgZmOZl8gXAF -O models/reranker/config_sentence_transformers.json
gdown 1chu947uQRrM2Y-CxKTbeR4Ln_CMpk1vT -O models/reranker/modules.json
gdown 1VPOC18m0gynFzBfkbeaE1VD_iN9XJHJb -O models/reranker/sentence_bert_config.json
gdown 1ED2XUOnJdia_QpZew9Qjjj7kPCqxBH0X -O models/reranker/special_tokens_map.json
gdown 1d5HjcPUkwgk76BU0BSErqCLG24KzeJ7P -O models/reranker/tokenizer_config.json
gdown 14awpQ0Bssu7T3O31Slmc_UxHay9xgyf0 -O models/reranker/tokenizer.json
gdown 1b1y9l7ELoKhqJxKKzlxxYxoGgHa1gwuQ -O models/reranker/vocab.txt
gdown 1Hz-4M153impq5f2r8iHUmnPzN9YB_Egl -O models/reranker/model.safetensors

echo "── Query classifier (DistilBERT fine-tuned, ~256 MB) ──"
gdown 1PxrQZWcor6Wgl0htuuIz4-unZu8lpzDN -O models/query_classifier/config.json
gdown 1hqbx7VQtkY0XF7GfYXfKoLiMFfBeuJKT -O models/query_classifier/special_tokens_map.json
gdown 1CY5bEYQd5L9WUVVDS1igvAe8HjR8K8r0 -O models/query_classifier/tokenizer_config.json
gdown 1MVGwvxWGSYg6-QAetnknromo82pxb9k- -O models/query_classifier/tokenizer.json
gdown 1YEZsrwU2VDkwDDUGi0WcmU-oofilSpOj -O models/query_classifier/vocab.txt
gdown 1pcDc8aulcaDpdVbShCPiwNsJVANvn30S -O models/query_classifier/training_args.bin
gdown 1qNo6ERtLS9LT63vQ4xZtgiwrxH0yVNPi -O models/query_classifier/model.safetensors

echo "── Verification ──"
python - <<'EOF'
from pathlib import Path
expected = {
    "models/reranker/model.safetensors": 90866412,
    "models/query_classifier/model.safetensors": 267835644,
}
ok = True
for p, size in expected.items():
    actual = Path(p).stat().st_size if Path(p).exists() else 0
    status = "OK" if actual == size else f"MISMATCH (got {actual})"
    if actual != size:
        ok = False
    print(f"{p}: {status}")
print("ALL WEIGHTS VERIFIED ✓" if ok else "!! SIZE MISMATCH — re-run the failed gdown line")
EOF
