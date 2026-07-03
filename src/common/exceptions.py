"""Exception hierarchy for VeritasRAG."""

class VeritasError(Exception):
    """Base exception for all VeritasRAG errors."""

class IngestionError(VeritasError):
    """Raised when an error occurs during document ingestion."""

class RetrievalError(VeritasError):
    """Raised when an error occurs during the retrieval stage."""

class GenerationError(VeritasError):
    """Raised when the LLM fails to generate an answer."""

class VerificationError(VeritasError):
    """Raised when the NLI verifier encounters an error."""
