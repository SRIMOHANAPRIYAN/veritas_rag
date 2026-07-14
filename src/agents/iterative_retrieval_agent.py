"""Iterative retrieval agent with multi-hop ReAct loop."""

import asyncio
from typing import List, Dict, Any, Protocol
from loguru import logger
import torch
import torch.nn.functional as F

from src.generation.llm_client import LlamaClient
from src.generation.model_registry import registry
from src.generation.prompt_templates import DECOMPOSITION_PROMPT, COVERAGE_CHECK_PROMPT
from src.generation.context_assembler import ContextAssembler

class NLIScorer(Protocol):
    """Protocol for NLI coverage check."""
    def verify(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Returns dict of entailment, contradiction, neutral probabilities."""
        ...

class ZeroShotDebertaNLIScorer:
    """Zero-shot NLI scorer using cross-encoder/nli-deberta-v3-base."""
    
    def verify(self, premise: str, hypothesis: str) -> Dict[str, float]:
        model, tokenizer = registry.get_nli_model()
        
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        labels = model.config.id2label
        result = {}
        for idx, prob in enumerate(probs):
            label = labels[idx].lower()
            if "entail" in label:
                result["entailment"] = float(prob)
            elif "contradict" in label:
                result["contradiction"] = float(prob)
            else:
                result["neutral"] = float(prob)
                
        if "entailment" not in result:
            result["contradiction"], result["entailment"], result["neutral"] = float(probs[0]), float(probs[1]), float(probs[2])
            
        return result

class IterativeRetrievalAgent:
    """Agent that performs multi-hop reasoning by decomposing and verifying."""
    
    def __init__(self, max_iterations: int = 3, max_sub_questions: int = 3):
        self.max_iterations = max_iterations
        self.max_sub_questions = max_sub_questions
        self.llm_client = LlamaClient()
        self.retriever = registry.get_hybrid_retriever()
        self.reranker = registry.get_reranker()
        self.nli_scorer = ZeroShotDebertaNLIScorer()
        self.context_assembler = ContextAssembler()
        self.entailment_threshold = 0.80

    async def decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into sub-queries."""
        prompt = DECOMPOSITION_PROMPT.format(query=query, max_sub_questions=self.max_sub_questions)
        response = await self.llm_client.generate(prompt)
        try:
            import json
            import re
            
            # Extract JSON from potential markdown formatting
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(response)
                
            sub_queries = data.get("sub_questions", [])
        except Exception as e:
            logger.error(f"Failed to parse JSON from decomposition: {response}. Error: {e}")
            sub_queries = []
            
        if not sub_queries:
            sub_queries = [query]
        return sub_queries

    async def retrieve_and_rerank(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve and rerank for a single sub-query."""
        # Using a thread to not block since retrieve/rerank are sync
        candidates = await asyncio.to_thread(self.retriever.retrieve, query)
        reranked = await asyncio.to_thread(self.reranker.rerank, query, candidates)
        return reranked

    async def run(self, query: str) -> List[Dict[str, Any]]:
        """Execute the multi-hop ReAct loop."""
        accumulated_chunks = {}
        
        # Step 1: Decompose
        sub_queries = await self.decompose_query(query)
        logger.info(f"Decomposed '{query}' into {len(sub_queries)} sub-queries: {sub_queries}")
        
        # Initial retrieval for sub-queries
        for sq in sub_queries:
            chunks = await self.retrieve_and_rerank(sq)
            for c in chunks:
                chunk_id = c["chunk"].chunk_id
                if chunk_id not in accumulated_chunks or c.get("rerank_score", 0) > accumulated_chunks[chunk_id].get("rerank_score", 0):
                    accumulated_chunks[chunk_id] = c
                        
        iteration = 1
        while iteration < self.max_iterations:
            context_str = self.context_assembler.assemble(query, list(accumulated_chunks.values()))
            
            # Formulate next step or check if fully answered
            # Generate a draft answer to verify
            draft_prompt = f"Using the context, answer the query: '{query}'. If you cannot answer it fully, reply with 'I cannot answer'."
            draft_answer = await self.llm_client.generate(draft_prompt)
            
            if "I cannot answer" not in draft_answer:
                # Use NLI to verify if the draft answer is actually entailed by the context
                nli_result = await asyncio.to_thread(self.nli_scorer.verify, context_str, draft_answer)
                if nli_result.get("entailment", 0) > self.entailment_threshold:
                    logger.info(f"NLI verified coverage (entailment={nli_result.get('entailment', 0):.2f}). Stopping.")
                    break
            
            # If not fully entailed or cannot answer, use coverage check prompt to get next query
            coverage_prompt = COVERAGE_CHECK_PROMPT.format(
                query=query, 
                context=context_str,
                draft_answer=draft_answer
            )
            next_action = await self.llm_client.generate(coverage_prompt)
            
            if "FULLY_ANSWERED" in next_action:
                logger.info("LLM claims fully answered despite NLI doubt. Stopping.")
                break
            else:
                logger.info(f"Coverage check requested more info: {next_action}")
                new_chunks = await self.retrieve_and_rerank(next_action)
                added = False
                for c in new_chunks:
                    chunk_id = c["chunk"].chunk_id
                    if chunk_id not in accumulated_chunks:
                        accumulated_chunks[chunk_id] = c
                        added = True
                
                if not added:
                    logger.warning("No new chunks found in iteration. Stopping.")
                    break
                    
            iteration += 1
        
        # Return ranked list of accumulated chunks (sorted by rerank score)
        final_chunks = sorted(list(accumulated_chunks.values()), key=lambda x: x.get("rerank_score", 0), reverse=True)
        return final_chunks
