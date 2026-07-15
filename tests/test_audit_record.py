import pytest
from datetime import datetime
from src.audit.audit_record import AuditRecord, ClaimVerdict, RetrievalCandidate

def test_audit_record_creation():
    candidate = RetrievalCandidate(chunk_id="chunk_1", rerank_score=0.95)
    
    verdict = ClaimVerdict(
        claim_text="The company was founded in 1990.",
        verdict="ENTAILED",
        evidence_chunk_id="chunk_1",
        evidence_span_start=10,
        evidence_span_end=42,
        confidence=0.88
    )
    
    record = AuditRecord(
        query_id="q123",
        query_text="When was the company founded?",
        retrieval_candidates=[candidate],
        chunks_used=["chunk_1"],
        answer_raw="The company was founded in 1990.",
        claims=[verdict],
        answer_final="The company was founded in 1990.",
        pipeline_latency_ms=1500
    )
    
    assert record.query_id == "q123"
    assert len(record.claims) == 1
    assert record.claims[0].verdict == "ENTAILED"
    assert record.claims[0].evidence_span_start == 10
    assert record.claims[0].evidence_span_end == 42
    assert record.contradiction_regenerations == 0
    assert record.multi_hop_triggered is False
    assert isinstance(record.timestamp, datetime)
