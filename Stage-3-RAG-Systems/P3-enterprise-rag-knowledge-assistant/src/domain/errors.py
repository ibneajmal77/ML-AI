class RAGError(Exception):
    """Base exception for all RAG system errors."""


class DocumentNotFoundError(RAGError):
    """Document ID not found in the database."""


class EmbeddingError(RAGError):
    """Azure OpenAI embedding call failed after all retries."""


class RerankerError(RAGError):
    """Cohere rerank call failed after all retries."""


class ContextBudgetExceededError(RAGError):
    """All retrieved chunks exceed the token budget."""


class AccessDeniedError(RAGError):
    """User clearance level is below the document's access level."""
