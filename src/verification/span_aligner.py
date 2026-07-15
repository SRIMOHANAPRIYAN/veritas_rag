from typing import Optional, Tuple
import re

class SpanAligner:
    """Finds the best evidence span for a claim within a chunk text."""
    
    @staticmethod
    def align(claim: str, chunk_text: str, tolerance: int = 5) -> Tuple[Optional[int], Optional[int]]:
        """
        Two-pointer character alignment. Returns (start_idx, end_idx) in chunk_text.
        O(n) approximate string matching.
        """
        if not claim or not chunk_text:
            return None, None
            
        # Normalize: lower and remove non-alphanumeric for matching
        def normalize(s: str):
            return re.sub(r'[^a-z0-9]', '', s.lower())
            
        claim_norm = normalize(claim)
        if not claim_norm:
            return None, None
            
        # We want to find a window in chunk_text that contains the characters of claim_norm in order
        # with minimal gaps. 
        # A true O(n) two-pointer approach:
        
        best_span = None
        min_length = float('inf')
        
        chunk_len = len(chunk_text)
        
        # To make it O(n) while finding best start, we can scan forward.
        # But for an exact or near-exact match, we can just do a simple greedy search.
        
        for i in range(chunk_len):
            # Check if this character matches the start of the claim
            if chunk_text[i].lower() == claim_norm[0] or (not chunk_text[i].isalnum()):
                if chunk_text[i].isalnum() and chunk_text[i].lower() != claim_norm[0]:
                    continue
                    
                # Potential start
                c_idx = 0
                j = i
                misses = 0
                
                start_match_idx = -1
                
                while j < chunk_len and c_idx < len(claim_norm):
                    char_j = chunk_text[j].lower()
                    
                    if not char_j.isalnum():
                        j += 1
                        continue
                        
                    if start_match_idx == -1 and char_j == claim_norm[c_idx]:
                        start_match_idx = j
                        
                    if char_j == claim_norm[c_idx]:
                        c_idx += 1
                    else:
                        misses += 1
                        if misses > tolerance:
                            break
                    j += 1
                    
                if c_idx == len(claim_norm):
                    span_len = j - start_match_idx
                    if span_len < min_length:
                        min_length = span_len
                        best_span = (start_match_idx, j)
                        
        return best_span[0] if best_span else None, best_span[1] if best_span else None

