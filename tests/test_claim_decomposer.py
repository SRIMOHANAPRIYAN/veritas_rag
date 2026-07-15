import pytest
from src.verification.claim_decomposer import ClaimDecomposer

def test_claim_decomposer():
    decomposer = ClaimDecomposer()
    
    # Test simple SVO
    text1 = "Apple acquired Beats Electronics."
    claims1 = decomposer.decompose(text1)
    assert len(claims1) == 1
    assert "Apple acquired Beats Electronics" in claims1[0] or claims1[0] == "Apple acquired Beats Electronics."
    
    # Test fallback
    text2 = "Yes."
    claims2 = decomposer.decompose(text2)
    assert len(claims2) == 1
    assert claims2[0] == "Yes."
    
    # Test multiple sentences
    text3 = "Google built a new headquarters. It is located in California."
    claims3 = decomposer.decompose(text3)
    assert len(claims3) == 2
    assert "Google built a new headquarters" in claims3[0] or claims3[0] == "Google built a new headquarters."
    assert "It is located in California" in claims3[1] or claims3[1] == "It is located in California."
