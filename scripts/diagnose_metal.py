"""Bisect the Metal crash: context size vs torch-MPS co-residency.

Run:  python scripts/diagnose_metal.py A   (Llama ALONE, n_ctx=8192, long prompt)
      python scripts/diagnose_metal.py B   (Llama ALONE, n_ctx=4096, long prompt)
      python scripts/diagnose_metal.py C   (Llama n_ctx=8192 WITH torch models on MPS)
Interpretation:
  A pass + C crash  -> co-residency/memory contention (fix: aux models to CPU and/or n_ctx=4096)
  A crash + B pass  -> large-ctx Metal issue          (fix: n_ctx=4096)
  A crash + B crash -> llama-cpp-python regression    (fix: pin/upgrade the wheel)
"""
import sys

from llama_cpp import Llama

MODE = sys.argv[1] if len(sys.argv) > 1 else "A"
N_CTX = 4096 if MODE == "B" else 8192

# ~2000-token prompt to mimic ask.py's assembled context
long_prompt = ("Section 4.2 Termination. Either party may terminate this agreement "
               "upon thirty days written notice provided that all outstanding fees "
               "are settled in full prior to the effective date. ") * 120

if MODE == "C":
    print(">> Loading torch models on MPS first (embedder + reranker + NLI)...")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    emb = SentenceTransformer("all-mpnet-base-v2", device="mps")
    rr = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="mps")
    nli = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/nli-deberta-v3-base").to("mps")
    _ = emb.encode(["warmup"])  # force real allocation
    print(">> torch models resident on MPS.")

print(f">> Loading Llama n_ctx={N_CTX} ...")
llm = Llama(model_path="models/llm/llama31-8b-q4.gguf",
            n_ctx=N_CTX, n_gpu_layers=-1, verbose=False)
print(">> Generating 256 tokens from ~2000-token prompt...")
out = llm(long_prompt + "\n\nSummarize the termination terms:", max_tokens=256)
print(">> SUCCESS. Output tail:", out["choices"][0]["text"][-120:])
print(f">> MODE {MODE} PASSED")
