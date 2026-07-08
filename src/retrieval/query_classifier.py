"""Query classifier with keyword-heuristic fallback."""

from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from loguru import logger
from src.common.device import DEVICE


class QueryClassifier:
    """Classifies queries into simple, multi-hop, or comparative.

    Includes a rule-based keyword fallback that overrides a 'simple'
    prediction when comparative/multi-hop cue phrases are detected.
    """

    def __init__(
        self,
        model_path: str,
        device: str = DEVICE,
        fallback_keywords: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.fallback_keywords = fallback_keywords or {}

        logger.info(f"Loading QueryClassifier from {model_path} on {device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(device)
            self.model.eval()
        except Exception as e:
            logger.warning(
                f"Query classifier model {model_path} could not be loaded. "
                f"Defaulting to 'simple'. Error: {e}"
            )
            self.model = None
            self.tokenizer = None

    def _keyword_override(self, query: str) -> Optional[str]:
        """Check if keyword heuristics should override a 'simple' prediction.

        Args:
            query: The user query string.

        Returns:
            'comparative' or 'multi-hop' if a keyword match is found,
            None otherwise.
        """
        query_lower = query.lower()

        # Check comparative keywords first (higher priority)
        for kw in self.fallback_keywords.get("comparative", []):
            if kw in query_lower:
                logger.debug(f"Keyword fallback: '{kw}' matched -> comparative")
                return "comparative"

        # Check multi-hop keywords
        for kw in self.fallback_keywords.get("multi_hop", []):
            if kw in query_lower:
                logger.debug(f"Keyword fallback: '{kw}' matched -> multi-hop")
                return "multi-hop"

        return None

    def classify(self, query: str) -> str:
        """Classify a query.

        Returns:
            str: 'simple', 'multi-hop', or 'comparative'.
        """
        if self.model is None or self.tokenizer is None:
            # Model not loaded - fall back to keyword heuristic or 'simple'
            override = self._keyword_override(query)
            return override if override else "simple"

        inputs = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class_id = logits.argmax().item()

        # The model was trained with 0=simple, 1=multi-hop, 2=comparative
        label = self.model.config.id2label.get(
            str(predicted_class_id),
            self.model.config.id2label.get(
                predicted_class_id, f"unknown_{predicted_class_id}"
            ),
        )

        # Apply keyword fallback: if model says 'simple' but keywords
        # suggest otherwise, override.
        if label == "simple":
            override = self._keyword_override(query)
            if override:
                logger.info(
                    f"Query '{query[:60]}...' model said 'simple', "
                    f"keyword fallback overrode to '{override}'"
                )
                return override

        logger.debug(f"Query '{query}' classified as {label}")
        return label
