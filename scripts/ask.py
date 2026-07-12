#!/usr/bin/env python3
"""Gate 3 end-to-end entrypoint: Agent retrieval + Generation with citations."""

import sys
import asyncio
import time
from loguru import logger
from omegaconf import OmegaConf

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.model_registry import registry
from src.generation.context_assembler import ContextAssembler
from src.generation.llm_client import LlamaClient
from src.generation.prompt_templates import GENERATION_PROMPT

async def async_main():
    if "--warm-check" in sys.argv:
        warm_check = True
        sys.argv.remove("--warm-check")
    else:
        warm_check = False
        
    if len(sys.argv) < 2 and not warm_check:
        print("Usage: python scripts/ask.py [--warm-check] 'Your query here'")
        sys.exit(1)
        
    query = " ".join(sys.argv[1:]) if not warm_check else ""
    
    if not warm_check:
        logger.info(f"Query: {query}")
    
    start_time = time.time()
    
    # Initialize components
    logger.info("Initializing components...")
    retriever = registry.get_hybrid_retriever()
    reranker = registry.get_reranker()
    assembler = ContextAssembler()
    llm = LlamaClient()
    
    if warm_check:
        logger.info("Warm check complete. Models are loaded.")
        print("\n[WARM-CHECK OK]")
        return
    
    # 1. Retrieve
    logger.info("Retrieving candidates...")
    candidates = retriever.retrieve(query)
    
    # 2. Rerank
    logger.info("Reranking candidates...")
    reranked = reranker.rerank(query, candidates)
    
    if not reranked:
        logger.warning("No chunks retrieved.")
        print("\nAnswer: I could not find any relevant information to answer your query.")
        return
        
    logger.info(f"Retrieved and reranked {len(reranked)} chunks.")
    
    # 3. Assemble Context
    context = assembler.assemble(query, reranked)
    
    # 4. Generate Final Answer
    logger.info("Generating final answer with citations...")
    prompt = GENERATION_PROMPT.format(context=context, query=query)
    
    answer = await llm.generate(prompt)
    
    end_time = time.time()
    
    print("\n" + "="*50)
    print("FINAL ANSWER")
    print("="*50)
    print(answer)
    print("="*50)
    print(f"Wall time: {end_time - start_time:.2f}s")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
