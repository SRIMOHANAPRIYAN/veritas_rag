from datetime import datetime, timezone
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class RetrievalCandidate(BaseModel):
    chunk_id: str
    faiss_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0

class ClaimVerdict(BaseModel):
    claim_text: str
    verdict: Literal["ENTAILED", "CONTRADICTED", "BASELESS"]
    evidence_chunk_id: Optional[str] = None
    evidence_span_start: Optional[int] = None
    evidence_span_end: Optional[int] = None
    confidence: float = 0.0

class AuditRecord(BaseModel):
    query_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query_text: str
    retrieval_candidates: List[RetrievalCandidate] = Field(default_factory=list)
    chunks_used: List[str] = Field(default_factory=list)
    answer_raw: str
    claims: List[ClaimVerdict] = Field(default_factory=list)
    answer_final: str
    contradiction_regenerations: int = 0
    pipeline_latency_ms: int = 0
    drift_score_at_query: float = 0.0
    multi_hop_triggered: bool = False
