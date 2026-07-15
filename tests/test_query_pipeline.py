import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.pipeline.query_pipeline import QueryPipeline
from src.audit.audit_record import AuditRecord, ClaimVerdict

@pytest.fixture
def mock_pipeline():
    with patch("src.pipeline.query_pipeline.registry") as mock_reg, \
         patch("src.pipeline.query_pipeline.ContextAssembler") as mock_asm, \
         patch("src.pipeline.query_pipeline.LlamaClient") as mock_llm_client, \
         patch("src.pipeline.query_pipeline.IterativeRetrievalAgent") as mock_agent, \
         patch("src.pipeline.query_pipeline.SelectiveRegenerator") as mock_regen:
         
        mock_reg.get_query_classifier.return_value = MagicMock()
        mock_reg.get_hybrid_retriever.return_value = MagicMock()
        mock_reg.get_reranker.return_value = MagicMock()
        
        yield mock_reg, mock_asm, mock_llm_client, mock_agent, mock_regen

@pytest.mark.anyio
async def test_query_pipeline_simple(mock_pipeline):
    mock_reg, mock_asm, mock_llm_client, mock_agent, mock_regen = mock_pipeline
    
    # Setup mocks
    classifier = mock_reg.get_query_classifier.return_value
    classifier.classify.return_value = "simple"
    
    retriever = mock_reg.get_hybrid_retriever.return_value
    retriever.retrieve.return_value = [{"chunk": MagicMock(chunk_id="c1", text="text1")}]
    
    reranker = mock_reg.get_reranker.return_value
    reranker.rerank.return_value = [{"chunk": MagicMock(chunk_id="c1", text="text1"), "rerank_score": 0.9}]
    
    llm = mock_llm_client.return_value
    llm.generate = AsyncMock(return_value="This is a simple answer.")
    
    regen = mock_regen.return_value
    regen.decomposer = MagicMock()
    regen.decomposer.decompose.return_value = ["This is a simple answer"]
    regen.verifier = MagicMock()
    regen.verifier.verify_claim.return_value = ClaimVerdict(claim_text="This is a simple answer", verdict="ENTAILED", confidence=0.99)
    regen.process_answer.return_value = ("This is a simple answer.", [], 0)
    
    pipeline = QueryPipeline()
    answer, record = await pipeline.run("What is simple?")
    
    assert answer == "This is a simple answer."
    assert isinstance(record, AuditRecord)
    assert record.multi_hop_triggered is False
    assert len(record.chunks_used) == 1
