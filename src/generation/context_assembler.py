"""Context assembler for Generation."""

from typing import List, Dict, Any
from omegaconf import OmegaConf
from loguru import logger

from src.generation.model_registry import registry
from src.generation.prompt_templates import SYSTEM_PROMPT


class ContextAssembler:
    """Assembles retrieved chunks into a prompt context, respecting the token budget."""

    def __init__(self):
        self.cfg = OmegaConf.load("configs/config.yaml")
        self.n_ctx = self.cfg.generation.get("n_ctx", 8192)
        self.max_tokens = self.cfg.generation.get("max_tokens", 512)
        # Margin for safety (e.g. user prompt overhead)
        self.margin = 100 
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens using the active LLM."""
        llm = registry.get_llm()
        # llama_cpp tokenize expects bytes
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False))

    def assemble(self, query: str, ranked_chunks: List[Dict[str, Any]]) -> str:
        """Assemble context from ranked chunks, dropping lowest-ranked if needed.
        
        Args:
            query: The user query (to account for its token cost).
            ranked_chunks: List of candidate chunk dicts, pre-sorted by relevance.
                           Must contain 'chunk' key with a Chunk object.
                           
        Returns:
            The assembled context string.
        """
        system_prompt_tokens = self._count_tokens(SYSTEM_PROMPT)
        query_tokens = self._count_tokens(query)
        
        # Calculate budget for the context string itself
        budget = self.n_ctx - system_prompt_tokens - self.max_tokens - query_tokens - self.margin
        
        if budget <= 0:
            logger.warning("Token budget is <= 0! Context will be empty.")
            return ""

        assembled_parts = []
        current_tokens = 0
        
        for candidate in ranked_chunks:
            chunk = candidate["chunk"]
            # Format: [CHUNK {id} | {doc} | p.{page}]\n{text}
            doc_id = chunk.doc_id
            page = chunk.page
            
            chunk_str = f"[CHUNK {chunk.chunk_id} | {doc_id} | p.{page}]\n{chunk.text}\n"
            chunk_tokens = self._count_tokens(chunk_str)
            
            if current_tokens + chunk_tokens <= budget:
                assembled_parts.append(chunk_str)
                current_tokens += chunk_tokens
            else:
                logger.debug(f"Truncated chunk {chunk.chunk_id} due to token budget.")
                break # Since chunks are ranked, drop the lowest ones
                
        context = "\n".join(assembled_parts).strip()
        logger.info(f"Assembled context with {len(assembled_parts)} chunks ({current_tokens} tokens).")
        return context
