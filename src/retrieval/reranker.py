"""Cross-encoder reranker with zero-shot / fine-tuned mode switch."""

from typing import Any, Dict, List

import torch
from sentence_transformers import CrossEncoder

from loguru import logger
from src.common.device import DEVICE


class CrossEncoderReranker:
    """Reranker using a CrossEncoder model.

    Supports two modes via config:
    - zeroshot: loads cross-encoder/ms-marco-MiniLM-L-6-v2
    - finetuned: loads models/reranker/
    """

    def __init__(
        self,
        model_path: str,
        top_k: int = 8,
        device: str = DEVICE,
    ) -> None:
        self.model_path = model_path
        self.top_k = top_k
        self.device = device

        logger.info(f"Loading CrossEncoder from {model_path} on {device}")
        try:
            self.model = CrossEncoder(model_path, device=device)
        except Exception as e:
            logger.warning(
                f"Reranker model {model_path} could not be loaded. "
                f"Defaulting to no reranker. Error: {e}"
            )
            self.model = None

    @classmethod
    def from_config(
        cls,
        mode: str,
        zeroshot_model: str,
        finetuned_model: str,
        top_k: int = 8,
        device: str = DEVICE,
    ) -> "CrossEncoderReranker":
        """Factory that selects model path based on config mode.

        Args:
            mode: 'zeroshot' or 'finetuned'.
            zeroshot_model: HuggingFace model name for zero-shot.
            finetuned_model: Local path for fine-tuned model.
            top_k: Number of results to return after reranking.
            device: Torch device string.

        Returns:
            Configured CrossEncoderReranker instance.
        """
        if mode == "finetuned":
            model_path = finetuned_model
        else:
            model_path = zeroshot_model
        logger.info(f"Reranker mode: {mode} -> {model_path}")
        return cls(model_path=model_path, top_k=top_k, device=device)

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank a list of candidate chunks for a query.

        Args:
            query: The user query string.
            candidates: List of dicts with at least a 'chunk' key
                        containing a Chunk object.

        Returns:
            List of dicts, sorted by reranker score, truncated to top_k.
            Adds 'rerank_score' to each dict.
        """
        if not candidates:
            return []

        if self.model is None:
            logger.warning("Reranker model not loaded. Returning original candidates.")
            return candidates[: self.top_k]

        # Prepare inputs: list of (query, chunk_text)
        model_inputs = [[query, c["chunk"].text] for c in candidates]

        # Predict
        scores = self.model.predict(model_inputs)

        # Add scores and sort
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(
                torch.sigmoid(torch.tensor(score)).item()
            )

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[: self.top_k]
