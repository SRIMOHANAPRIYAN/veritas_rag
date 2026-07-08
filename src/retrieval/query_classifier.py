import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from loguru import logger
from src.common.device import DEVICE

class QueryClassifier:
    """Classifies queries into simple, multi-hop, or comparative."""
    
    def __init__(self, model_path: str, device: str = DEVICE):
        self.model_path = model_path
        self.device = device
        
        logger.info(f"Loading QueryClassifier from {model_path} on {device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Query classifier model {model_path} could not be loaded. Defaulting to 'simple'. Error: {e}")
            self.model = None
            self.tokenizer = None
            
    def classify(self, query: str) -> str:
        """Classify a query.
        
        Returns:
            str: "simple", "multi-hop", or "comparative"
        """
        if self.model is None or self.tokenizer is None:
            return "simple"
            
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        
        # The model was trained with 0=simple, 1=multi-hop, 2=comparative
        label = self.model.config.id2label[predicted_class_id]
        
        logger.debug(f"Query '{query}' classified as {label}")
        return label
