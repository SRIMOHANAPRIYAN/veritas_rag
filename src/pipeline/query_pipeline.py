import time
import asyncio
from typing import Tuple, List, Dict, Any
from datetime import datetime, timezone
from loguru import logger
from pydantic import ValidationError

from src.generation.model_registry import registry
from src.generation.context_assembler import ContextAssembler
from src.generation.llm_client import LlamaClient
from src.generation.prompt_templates import GENERATION_PROMPT
from src.agents.iterative_retrieval_agent import IterativeRetrievalAgent
from src.verification.selective_regen import SelectiveRegenerator
from src.audit.audit_record import AuditRecord, RetrievalCandidate, ClaimVerdict

class QueryPipeline:
    def __init__(self):
        self.classifier = registry.get_query_classifier()
        self.retriever = registry.get_hybrid_retriever()
        self.reranker = registry.get_reranker()
        self.assembler = ContextAssembler()
        self.llm = LlamaClient()
        self.agent = IterativeRetrievalAgent(
            retriever=self.retriever,
            reranker=self.reranker,
            llm_client=self.llm,
            context_assembler=self.assembler,
        )
        self.regenerator = SelectiveRegenerator()
        
    def _validate(self, query: str) -> bool:
        """Validate the query."""
        if not query or not query.strip():
            return False
        # Basic prompt injection guard (if any) could go here
        if len(query) > 1000: # arbitrary max length
            return False
        return True

    async def run(self, query: str) -> Tuple[str, AuditRecord]:
        """
        Runs the full RAG pipeline with verification and selective regeneration.
        Returns the final answer and its AuditRecord.
        """
        start_time = time.time()
        
        # 1. Validate
        if not self._validate(query):
            record = AuditRecord(
                query_id=f"q_{int(time.time()*1000)}",
                query_text=query,
                answer_raw="Invalid query.",
                answer_final="Invalid query."
            )
            return "Invalid query.", record
            
        # 2. Classify
        classification = self.classifier.classify(query)
        multi_hop = (classification == "multi-hop")
        
        # 3-5. Retrieve & Rerank (or Agent)
        if multi_hop:
            logger.info("Multi-hop query detected. Engaging IterativeRetrievalAgent.")
            chunks = await self.agent.run(query)
        else:
            logger.info("Simple/Comparative query detected. Standard retrieval.")
            candidates = self.retriever.retrieve(query)
            chunks = self.reranker.rerank(query, candidates)
            
        # Extract retrieval candidates for audit
        audit_candidates = []
        for c in chunks:
            audit_candidates.append(
                RetrievalCandidate(
                    chunk_id=c["chunk"].chunk_id,
                    faiss_score=c.get("faiss_score", 0.0),
                    bm25_score=c.get("bm25_score", 0.0),
                    rrf_score=c.get("rrf_score", 0.0),
                    rerank_score=c.get("rerank_score", 0.0)
                )
            )
            
        chunks_used = [c["chunk"].chunk_id for c in chunks]
        
        # 6. Assemble
        context = self.assembler.assemble(query, chunks)
        
        # Build chunk text map for the verification step
        chunk_text_map = {c["chunk"].chunk_id: c["chunk"].text for c in chunks}
        
        # 7. Generate
        prompt = GENERATION_PROMPT.format(context=context, query=query)
        raw_answer = await self.llm.generate(prompt)
        
        if not chunks:
            raw_answer = "I could not find any relevant information to answer your query."
            record = AuditRecord(
                query_id=f"q_{int(time.time()*1000)}",
                query_text=query,
                answer_raw=raw_answer,
                answer_final=raw_answer,
                multi_hop_triggered=multi_hop,
                pipeline_latency_ms=int((time.time() - start_time) * 1000)
            )
            return raw_answer, record
            
        # 8. Decompose & Verify
        initial_claims_texts = self.regenerator.decomposer.decompose(raw_answer)
        initial_verdicts = []
        
        for claim_text in initial_claims_texts:
            # We must test against all retrieved chunks to find the best support
            # For simplicity, we can concatenate or just test the one with highest entailment
            best_verdict = None
            for chunk_id, text in chunk_text_map.items():
                verdict = self.regenerator.verifier.verify_claim(claim_text, text, chunk_id)
                if not best_verdict or verdict.verdict == "ENTAILED" or (verdict.verdict == "BASELESS" and best_verdict.verdict == "CONTRADICTED"):
                    best_verdict = verdict
                if best_verdict.verdict == "ENTAILED":
                    break
            
            if best_verdict:
                initial_verdicts.append(best_verdict)
                
        # 9. Selective Regeneration
        final_answer, final_claims, regenerations = self.regenerator.process_answer(
            raw_answer, initial_verdicts, chunk_text_map, query
        )
        
        end_time = time.time()
        
        # Build Audit Record
        record = AuditRecord(
            query_id=f"q_{int(end_time*1000)}",
            timestamp=datetime.now(timezone.utc),
            query_text=query,
            retrieval_candidates=audit_candidates,
            chunks_used=chunks_used,
            answer_raw=raw_answer,
            claims=final_claims,
            answer_final=final_answer,
            contradiction_regenerations=regenerations,
            pipeline_latency_ms=int((end_time - start_time) * 1000),
            multi_hop_triggered=multi_hop
        )
        
        return final_answer, record
