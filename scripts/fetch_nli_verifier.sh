#!/usr/bin/env bash
# Fetch the fine-tuned NLI verifier from Google Drive (file IDs resolved 2026-07-15).
# Downloads to models/nli_verifier/. Excludes checkpoint-*/.
# Usage: bash scripts/fetch_nli_verifier.sh   (run from repo root, venv active)
set -euo pipefail

command -v gdown >/dev/null 2>&1 || pip install gdown

mkdir -p models/nli_verifier

echo "── NLI verifier (deberta-v3-base domain fine-tune, ~738 MB) ──"
gdown 1rFdwIY75Pp9fRDhqwiyaXTy2UYa4zDd7 -O models/nli_verifier/config.json
gdown 1f0ylnqjfOU7DWOUnVqVrO8ficGNOC047 -O models/nli_verifier/tokenizer_config.json
gdown 19HS9l_3Sx1cj7EhynqT2edMyChpTXmXp -O models/nli_verifier/special_tokens_map.json
gdown 1_x6TdEfk3wzpWE1aJf2Q5dTfcKXqNFuT -O models/nli_verifier/added_tokens.json
gdown 1PPbsx1BJ9rK89P-M3PNKSJ31ZQAClcS- -O models/nli_verifier/spm.model
gdown 1HY0Gt3Ij0aUMpJeQ_qj5r1RtuUB00H57 -O models/nli_verifier/tokenizer.json
gdown 1TW83QkSlNxOzsfgFsd3udwNQSyRZ7PPy -O models/nli_verifier/training_args.bin
gdown 1-u_No8m6bQm-93jDKOxQn_pMFtPaclHL -O models/nli_verifier/model.safetensors

echo "── Verification ──"
python - <<'EOF'
from pathlib import Path
p = "models/nli_verifier/model.safetensors"
size = Path(p).stat().st_size if Path(p).exists() else 0
print(f"{p}: {'OK' if size == 737722356 else f'MISMATCH (got {size})'}")
print("NLI VERIFIER VERIFIED" if size == 737722356 else "!! re-run the failed gdown line")
EOF
