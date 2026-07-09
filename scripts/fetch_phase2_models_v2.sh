#!/usr/bin/env bash
# Fetch Phase 2 V2 trained models from Google Drive (file IDs resolved 2026-07-09).
# Downloads to models/reranker_v2/ and models/query_classifier_v2/ — v1 folders untouched.
# Usage: bash scripts/fetch_phase2_models_v2.sh   (run from repo root, venv active)
set -euo pipefail

command -v gdown >/dev/null 2>&1 || pip install gdown

mkdir -p models/reranker_v2 models/query_classifier_v2

echo "── Reranker v2 (MS MARCO-mixed, ~87 MB) ──"
gdown 1CDG1xL8fAPpdCrEkdOf_rm0w9YLS-wKe -O models/reranker_v2/config.json
gdown 10QTn5ZzGSsYpP8qihD9i3fNZ_H-qqvmx -O models/reranker_v2/config_sentence_transformers.json
gdown 1EJxIv3eGpNzqrpxUFTbJpquWSaWRb-M1 -O models/reranker_v2/modules.json
gdown 1dJhxnF680wbOY3wp_FFmCNmOgCJA1bjh -O models/reranker_v2/sentence_bert_config.json
gdown 1V-xtLnKW7_tn-C5uRHVO0MOtabhMq3bk -O models/reranker_v2/special_tokens_map.json
gdown 1aW8XcjHcUQWe6H5R_TyGCfdjz11Lx4Aj -O models/reranker_v2/tokenizer_config.json
gdown 1WQbqf6dNAiO1WgbWQrjvrmHy7hWygc70 -O models/reranker_v2/tokenizer.json
gdown 1AsL9Sx1ixytDm61qUFVZAevYpWU49qSH -O models/reranker_v2/vocab.txt
gdown 1Pqsqge_gEtimy1dGwRO1tCxlY1SlmEL7 -O models/reranker_v2/model.safetensors

echo "── Query classifier v2 (legal-domain mixed, ~256 MB) ──"
gdown 15WwkJO-RF5AB1hpOXshvd4ecXJXnLAwm -O models/query_classifier_v2/config.json
gdown 1wnRHjGrFUZwxJaY39gTVBhyNgldpKHf3 -O models/query_classifier_v2/special_tokens_map.json
gdown 1fEBvRCZkLydv9fl3CekaxLczoLiQXymN -O models/query_classifier_v2/tokenizer_config.json
gdown 1Uk81jGBifAwDs-S9ILqvvtRpy40wr1JR -O models/query_classifier_v2/tokenizer.json
gdown 1mp3Jg8vjRqVEF6W-hKSkIcP4Dlynze0R -O models/query_classifier_v2/vocab.txt
gdown 17yv-mizhm5zj0yg_JXNmOHX_IVm6Jxsn -O models/query_classifier_v2/training_args.bin
gdown 1T4UDhd5UPhPj4dvHQF-VPQ3j2erYdd9Z -O models/query_classifier_v2/model.safetensors

echo "── Verification ──"
python - <<'EOF'
from pathlib import Path
expected = {
    "models/reranker_v2/model.safetensors": 90866412,
    "models/query_classifier_v2/model.safetensors": 267835644,
}
ok = True
for p, size in expected.items():
    actual = Path(p).stat().st_size if Path(p).exists() else 0
    print(f"{p}: {'OK' if actual == size else f'MISMATCH (got {actual})'}")
    ok = ok and actual == size
print("ALL V2 WEIGHTS VERIFIED ✓" if ok else "!! MISMATCH — re-run the failed gdown line")
EOF
