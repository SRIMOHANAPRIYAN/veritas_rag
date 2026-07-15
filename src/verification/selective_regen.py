from typing import List, Tuple
import spacy
from loguru import logger

from src.generation.model_registry import registry
from src.generation.prompt_templates import REGEN_PROMPT
from src.audit.audit_record import ClaimVerdict
from src.verification.claim_decomposer import ClaimDecomposer
from src.verification.nli_verifier import NLIVerifier

class SelectiveRegenerator:
    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts
        self.decomposer = ClaimDecomposer()
        self.verifier = NLIVerifier()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def regenerate_sentence(self, sentence: str, evidence_text: str, evidence_id: str, question: str) -> Tuple[str, List[ClaimVerdict], bool]:
        """
        Regenerates a contradicted sentence.
        Returns (new_sentence, final_claims, is_unresolved).
        """
        llm = registry.get_llm()
        
        for attempt in range(self.max_attempts):
            prompt = REGEN_PROMPT.format(
                question=question,
                sentence=sentence,
                evidence=evidence_text
            )
            
            response = llm(
                prompt,
                max_tokens=100,
                stop=["\n", "Original", "Rewrite"],
                echo=False
            )
            new_sentence = response["choices"][0]["text"].strip()
            
            # If the model gives up
            if "cannot answer" in new_sentence.lower():
                # Flag as unresolved, returning the empty or generic string
                return new_sentence, [], True
                
            # Re-run decomposition and verification
            new_claims_text = self.decomposer.decompose(new_sentence)
            new_verdicts = []
            all_clear = True
            
            for claim_text in new_claims_text:
                verdict = self.verifier.verify_claim(claim_text, evidence_text, evidence_id)
                new_verdicts.append(verdict)
                if verdict.verdict == "CONTRADICTED":
                    all_clear = False
                    
            if all_clear:
                return new_sentence, new_verdicts, False
                
        # If we exhausted attempts and it's still contradicted
        # Flag as UNRESOLVED
        for v in new_verdicts:
            if v.verdict == "CONTRADICTED":
                v.verdict = "UNRESOLVED"
                
        return new_sentence, new_verdicts, True

    def process_answer(self, raw_answer: str, claims: List[ClaimVerdict], chunk_text_map: dict, question: str) -> Tuple[str, List[ClaimVerdict], int]:
        """
        Processes a full answer. If any claims are CONTRADICTED, finds their parent sentence,
        and regenerates it.
        Returns (final_answer, final_claims, num_regenerations).
        """
        doc = self.nlp(raw_answer)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        final_claims = []
        final_sentences = []
        num_regenerations = 0
        
        # Mapping sentences to their claims
        # For simplicity, we assume a claim maps to the sentence that contains it.
        for sent_idx, sentence in enumerate(sentences):
            sent_claims = [c for c in claims if c.claim_text in sentence or c.claim_text.strip('.') in sentence]
            
            needs_regen = False
            contradicted_claim = None
            for c in sent_claims:
                if c.verdict == "CONTRADICTED":
                    needs_regen = True
                    contradicted_claim = c
                    break
                    
            if needs_regen and contradicted_claim and contradicted_claim.evidence_chunk_id:
                evidence_text = chunk_text_map.get(contradicted_claim.evidence_chunk_id, "")
                if evidence_text:
                    new_sentence, new_verdicts, unresolved = self.regenerate_sentence(
                        sentence, evidence_text, contradicted_claim.evidence_chunk_id, question
                    )
                    final_sentences.append(new_sentence)
                    final_claims.extend(new_verdicts)
                    num_regenerations += 1
                else:
                    final_sentences.append(sentence)
                    final_claims.extend(sent_claims)
            else:
                final_sentences.append(sentence)
                final_claims.extend(sent_claims)
                
        # Any claims that didn't map to a sentence (should be rare)
        mapped_claim_texts = {c.claim_text for c in final_claims}
        for c in claims:
            if c.claim_text not in mapped_claim_texts and c.verdict != "CONTRADICTED":
                final_claims.append(c)
                
        final_answer = " ".join(final_sentences)
        return final_answer, final_claims, num_regenerations
