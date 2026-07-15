"""Lazy-loading singleton registry for heavyweight models."""

import asyncio
from typing import Optional, Any, Tuple
from pathlib import Path
from omegaconf import OmegaConf
from loguru import logger
import threading
import concurrent.futures

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.query_classifier import QueryClassifier
from src.common.device import DEVICE

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    logger.warning("llama-cpp-python not installed. LLM features will be disabled.")

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelRegistry:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(ModelRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.cfg = OmegaConf.load("configs/config.yaml")
        
        self._llm = None
        self._hybrid_retriever = None
        self._reranker = None
        self._nli_model = None
        self._nli_tokenizer = None
        self._classifier = None

        # ThreadPoolExecutor to ensure all llama.cpp calls happen on a single OS thread
        self.llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        
        self._initialized = True

    def get_llm(self) -> Any:
        if self._llm is None:
            if Llama is None:
                raise ImportError("llama-cpp-python is required for LLM")
            model_path = self.cfg.generation.model_path
            n_ctx = self.cfg.generation.get("n_ctx", 8192)
            device = self.cfg.generation.get("device", DEVICE)
            
            logger.info(f"Loading LLM from {model_path} with n_ctx={n_ctx} on {device}")
            # Llama metal requires n_gpu_layers=-1 to use GPU
            n_gpu_layers = -1 if device == "mps" else 0
            
            self._llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
        return self._llm

    def get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            mode = self.cfg.reranker.mode
            zeroshot_model = self.cfg.reranker.zeroshot_model
            finetuned_model = self.cfg.reranker.finetuned_model
            top_k = self.cfg.reranker.top_k_reranked
            self._reranker = CrossEncoderReranker.from_config(
                mode=mode,
                zeroshot_model=zeroshot_model,
                finetuned_model=finetuned_model,
                top_k=top_k,
                device=DEVICE
            )
        return self._reranker

    def get_hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            faiss_path = self.cfg.indexer.faiss_index_path
            bm25_path = self.cfg.indexer.bm25_index_path
            metadata_db_path = self.cfg.indexer.metadata_db_path
            manifest_path = str(Path(faiss_path).parent / "manifest.json")
            model_name = self.cfg.indexer.embedding_model
            rrf_k = self.cfg.retriever.rrf_k
            top_k = self.cfg.retriever.top_k_fused
            device = self.cfg.generation.get("device", DEVICE)
            
            self._hybrid_retriever = HybridRetriever(
                faiss_path=faiss_path,
                bm25_path=bm25_path,
                metadata_db_path=metadata_db_path,
                manifest_path=manifest_path,
                model_name=model_name,
                rrf_k=rrf_k,
                top_k=top_k,
                device=device
            )
        return self._hybrid_retriever

    def get_nli_model(self) -> Tuple[Any, Any]:
        if self._nli_model is None:
            # Try local fine-tuned path first, fallback to base model
            local_path = self.cfg.verification.get("nli_model_path", "models/nli_verifier/")
            if Path(local_path).exists():
                model_name = local_path
            else:
                model_name = "cross-encoder/nli-deberta-v3-base"
            device = self.cfg.verification.get("device", DEVICE)
            logger.info(f"Loading NLI model {model_name} on {device}")
            self._nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            self._nli_model.eval()
        return self._nli_model, self._nli_tokenizer

    def get_query_classifier(self) -> QueryClassifier:
        if self._classifier is None:
            model_path = self.cfg.classifier.model_path
            fallback_keywords = self.cfg.classifier.get("fallback_keywords", {})
            self._classifier = QueryClassifier(
                model_path=model_path,
                device=DEVICE,
                fallback_keywords=fallback_keywords
            )
        return self._classifier

# Global singleton instance
registry = ModelRegistry()
