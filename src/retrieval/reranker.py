import os
from typing import List, Dict, Any
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from loguru import logger
from src.ingestion.metadata_tagger import Chunk
from src.common.device import DEVICE

class CrossEncoderReranker:
    """Reranker using a fine-tuned CrossEncoder model."""
    
    def __init__(self, model_path: str, top_k: int = 8, device: str = DEVICE):
        self.model_path = model_path
        self.top_k = top_k
        self.device = device
        
        logger.info(f"Loading CrossEncoder from {model_path} on {device}")
        try:
            self.model = CrossEncoder(model_path, device=device)
        except Exception as e:
            logger.warning(f"Reranker model {model_path} could not be loaded. Defaulting to no reranker. Error: {e}")
            self.model = None
            
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank a list of candidate chunks for a query.
        
        Args:
            query: The user query string.
            candidates: List of dicts with at least a "chunk" key containing a Chunk object.
            
        Returns:
            List of dicts, sorted by reranker score, truncated to top_k.
            Adds 'rerank_score' to each dict.
        """
        if not candidates:
            return []
            
        if self.model is None:
            logger.warning("Reranker model not loaded. Returning original candidates.")
            return candidates[:self.top_k]
            
        # Prepare inputs: list of (query, chunk_text)
        model_inputs = [[query, c["chunk"].text] for c in candidates]
        
        # Predict
        scores = self.model.predict(model_inputs)
        
        # Add scores and sort
        for idx, score in enumerate(scores):
            # Sigmoid to convert logit to 0-1 probability
            candidates[idx]["rerank_score"] = float(torch.sigmoid(torch.tensor(score)).item())
            
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:self.top_k]
