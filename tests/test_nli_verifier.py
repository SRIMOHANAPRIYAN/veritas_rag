import pytest
from unittest.mock import patch, MagicMock

from src.verification.nli_verifier import NLIVerifier
from src.audit.audit_record import ClaimVerdict

@pytest.fixture
def mock_registry():
    with patch("src.verification.nli_verifier.registry") as mock_reg:
        # We don't actually need the model to return real tensors for the unit test of verify_claim
        # because we can just mock verify()
        yield mock_reg

def test_nli_verifier_entailed(mock_registry):
    verifier = NLIVerifier()
    verifier.entail_threshold = 0.80
    verifier.contra_threshold = 0.70
    
    with patch.object(verifier, 'verify', return_value={"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04}):
        verdict = verifier.verify_claim("Apple makes iPhones", "Apple is a tech company that makes iPhones.", "chunk_1")
        
        assert verdict.verdict == "ENTAILED"
        assert verdict.confidence == 0.95
        assert verdict.evidence_chunk_id == "chunk_1"

def test_nli_verifier_contradicted(mock_registry):
    verifier = NLIVerifier()
    verifier.entail_threshold = 0.80
    verifier.contra_threshold = 0.70
    
    with patch.object(verifier, 'verify', return_value={"entailment": 0.05, "contradiction": 0.85, "neutral": 0.10}):
        verdict = verifier.verify_claim("Apple makes Android phones", "Apple is a tech company that makes iPhones.", "chunk_1")
        
        assert verdict.verdict == "CONTRADICTED"
        assert verdict.confidence == 0.85

def test_nli_verifier_baseless(mock_registry):
    verifier = NLIVerifier()
    verifier.entail_threshold = 0.80
    verifier.contra_threshold = 0.70
    
    with patch.object(verifier, 'verify', return_value={"entailment": 0.40, "contradiction": 0.40, "neutral": 0.20}):
        verdict = verifier.verify_claim("Apple sells coffee", "Apple is a tech company that makes iPhones.", "chunk_1")
        
        assert verdict.verdict == "BASELESS"
        assert verdict.confidence == 0.20
