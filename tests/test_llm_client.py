"""Regression tests for LlamaClient concurrency fixes."""

import pytest
import asyncio
from src.generation.llm_client import LlamaClient

@pytest.mark.local_model
def test_llm_client_sequential_generate():
    """Ensure that sequential LLM calls do not cause Metal context crashes.
    
    Submits 10 calls to the executor back-to-back.
    """
    
    async def run_test():
        client = LlamaClient()
        
        # Use a simple prompt to minimize generation time
        prompt = "Reply with 'OK'."
        
        for i in range(10):
            response = await client.generate(prompt=prompt, system_prompt="You are a helpful assistant.")
            assert response is not None
            assert len(response) > 0

    asyncio.run(run_test())

