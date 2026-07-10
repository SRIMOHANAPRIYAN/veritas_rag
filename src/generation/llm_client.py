"""LLM Client for generation and reasoning."""

import time
import asyncio
from typing import Optional
from loguru import logger
from omegaconf import OmegaConf

from src.common.exceptions import GenerationError
from src.generation.model_registry import registry
from src.generation.prompt_templates import SYSTEM_PROMPT


class LlamaClient:
    """Client for querying the LLM via llama-cpp-python."""

    def __init__(self):
        self.cfg = OmegaConf.load("configs/config.yaml")
        
        # Get defaults from config
        self.max_tokens = self.cfg.generation.get("max_tokens", 512)
        self.temperature = self.cfg.generation.get("temperature", 0.0)
        self.n_ctx = self.cfg.generation.get("n_ctx", 8192)
        self.timeout_warn_s = self.cfg.generation.get("timeout_warn_s", 90.0)
        
    async def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Generate a response using the LLM.
        
        Args:
            prompt: The user prompt.
            system_prompt: The system prompt to use.
            
        Returns:
            The generated response string.
            
        Raises:
            GenerationError: If the generation fails.
        """
        try:
            start_time = time.monotonic()
            
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                registry.llm_executor,
                self._sync_generate, prompt, system_prompt
            )
            
            elapsed = time.monotonic() - start_time
            if elapsed > self.timeout_warn_s:
                logger.warning(f"LLM generation took {elapsed:.2f}s (exceeded warning threshold {self.timeout_warn_s}s)")
                
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise GenerationError(f"LLM generation failed: {e}") from e

    def _sync_generate(self, prompt: str, system_prompt: str) -> str:
        """Synchronous call to llama.cpp."""
        llm = registry.get_llm()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        try:
            return output["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise GenerationError(f"Unexpected LLM output format: {output}") from e
