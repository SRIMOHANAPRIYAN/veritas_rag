import spacy
from typing import List

class ClaimDecomposer:
    """Extracts atomic claims (Subject-Verb-Object triples) from text."""
    
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)

    def _get_subject(self, token) -> str:
        """Find the subject of a given verb token."""
        for child in token.lefts:
            if child.dep_ in ("nsubj", "nsubjpass", "expl", "csubj"):
                # get full subtree for the subject
                return " ".join([t.text for t in child.subtree]).strip()
        return ""
        
    def _get_object(self, token) -> str:
        """Find the object of a given verb token."""
        for child in token.rights:
            if child.dep_ in ("dobj", "pobj", "attr", "acomp", "ccomp"):
                return " ".join([t.text for t in child.subtree]).strip()
        # sometimes object is behind a preposition
        for child in token.rights:
            if child.dep_ == "prep":
                return " ".join([t.text for t in child.subtree]).strip()
        return ""

    def decompose(self, text: str) -> List[str]:
        """Decompose text into atomic claim strings."""
        doc = self.nlp(text)
        claims = []
        
        for sent in doc.sents:
            # simple heuristic: find the root verb
            root = sent.root
            if root.pos_ == "VERB" or root.pos_ == "AUX":
                subject = self._get_subject(root)
                obj = self._get_object(root)
                
                # If we missed subject/object due to complex tree, just use the whole sentence as a fallback claim
                if subject and obj:
                    verb = root.text
                    
                    # collect auxiliaries and negation that come before the verb
                    pre_verb = []
                    for child in root.lefts:
                        if child.dep_ in ("aux", "auxpass", "neg"):
                            pre_verb.append(child.text)
                            
                    verb_phrase = " ".join(pre_verb + [verb]) if pre_verb else verb
                    
                    claim = f"{subject} {verb_phrase} {obj}"
                    claims.append(claim)
                else:
                    claims.append(sent.text.strip())
            else:
                # If root is not a verb, fallback to sentence
                claims.append(sent.text.strip())
                
        # deduplicate while preserving order
        seen = set()
        unique_claims = []
        for claim in claims:
            if claim not in seen:
                seen.add(claim)
                unique_claims.append(claim)
                
        return unique_claims
