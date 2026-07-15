import pytest
from unittest.mock import patch, MagicMock
from src.audit.audit_record import ClaimVerdict
from src.verification.selective_regen import SelectiveRegenerator

@pytest.fixture
def mock_llm_registry():
    with patch("src.verification.selective_regen.registry") as mock_reg:
        mock_llm = MagicMock()
        mock_reg.get_llm.return_value = mock_llm
        yield mock_llm

def test_selective_regenerator_success(mock_llm_registry):
    mock_llm_registry.return_value = {"choices": [{"text": "Apple acquired Beats in 2014."}]}
    
    regenerator = SelectiveRegenerator()
    
    # Mock decomposer and verifier
    regenerator.decomposer = MagicMock()
    regenerator.decomposer.decompose.return_value = ["Apple acquired Beats in 2014"]
    
    regenerator.verifier = MagicMock()
    verdict = ClaimVerdict(claim_text="Apple acquired Beats in 2014", verdict="ENTAILED", evidence_chunk_id="c1", confidence=0.99)
    regenerator.verifier.verify_claim.return_value = verdict
    
    new_sent, new_claims, unresolved = regenerator.regenerate_sentence(
        "Apple bought Google in 2014.",
        "Apple acquired Beats Electronics in 2014.",
        "c1",
        "Who did Apple acquire in 2014?"
    )
    
    assert unresolved is False
    assert new_sent == "Apple acquired Beats in 2014."
    assert len(new_claims) == 1
    assert new_claims[0].verdict == "ENTAILED"

def test_selective_regenerator_unresolved(mock_llm_registry):
    mock_llm_registry.return_value = {"choices": [{"text": "Apple bought Google in 2014."}]}
    
    regenerator = SelectiveRegenerator()
    regenerator.decomposer = MagicMock()
    regenerator.decomposer.decompose.return_value = ["Apple bought Google in 2014"]
    
    regenerator.verifier = MagicMock()
    verdict = ClaimVerdict(claim_text="Apple bought Google in 2014", verdict="CONTRADICTED", evidence_chunk_id="c1", confidence=0.88)
    regenerator.verifier.verify_claim.return_value = verdict
    
    new_sent, new_claims, unresolved = regenerator.regenerate_sentence(
        "Apple bought Google in 2014.",
        "Apple acquired Beats Electronics in 2014.",
        "c1",
        "Who did Apple acquire in 2014?"
    )
    
    assert unresolved is True
    # It should have updated the verdict to UNRESOLVED after failing max_attempts
    assert new_claims[0].verdict == "UNRESOLVED"
