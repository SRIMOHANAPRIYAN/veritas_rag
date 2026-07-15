from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.generation.model_registry import registry
from src.audit.audit_record import ClaimVerdict
from src.verification.span_aligner import SpanAligner

class NLIVerifier:
    """NLI Verifier using fine-tuned DeBERTa model."""
    
    def __init__(self):
        self.cfg = OmegaConf.load("configs/config.yaml")
        self.entail_threshold = self.cfg.verification.get("entailment_threshold", 0.80)
        self.contra_threshold = self.cfg.verification.get("contradiction_threshold", 0.70)
        
    def verify(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Returns dict of entailment, contradiction, neutral probabilities."""
        model, tokenizer = registry.get_nli_model()
        
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        labels = model.config.id2label
        result = {}
        for idx, prob in enumerate(probs):
            label = labels[idx].lower()
            if "entail" in label:
                result["entailment"] = float(prob)
            elif "contradict" in label:
                result["contradiction"] = float(prob)
            else:
                result["neutral"] = float(prob)
                
        return result
        
    def verify_claim(self, claim: str, chunk_text: str, chunk_id: str) -> ClaimVerdict:
        """
        Verifies a single claim against an evidence chunk.
        Returns ENTAILED if entailment > entail_threshold.
        Returns CONTRADICTED if contradiction > contra_threshold AND entailment < 0.20.
        Otherwise BASELESS.
        """
        probs = self.verify(premise=chunk_text, hypothesis=claim)
        
        entail = probs.get("entailment", 0.0)
        contra = probs.get("contradiction", 0.0)
        
        if entail > self.entail_threshold:
            verdict_str = "ENTAILED"
            confidence = entail
        elif contra > self.contra_threshold and entail < 0.20:
            verdict_str = "CONTRADICTED"
            confidence = contra
        else:
            verdict_str = "BASELESS"
            confidence = probs.get("neutral", 0.0)
            
        start, end = SpanAligner.align(claim, chunk_text)
        
        return ClaimVerdict(
            claim_text=claim,
            verdict=verdict_str,
            evidence_chunk_id=chunk_id,
            evidence_span_start=start,
            evidence_span_end=end,
            confidence=confidence
        )
