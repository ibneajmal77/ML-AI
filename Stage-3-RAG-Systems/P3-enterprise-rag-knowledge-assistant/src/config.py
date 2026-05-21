from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    azure_openai_api_version: str = "2024-10-21"

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Cohere
    cohere_api_key: str

    # Langfuse — empty string = tracing disabled (NullSpan pattern)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Azure Document Intelligence
    azure_di_endpoint: str = ""
    azure_di_key: str = ""

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Retrieval
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072
    reranker_top_k: int = 20
    final_top_k: int = 5
    chunk_size: int = 512
    parent_chunk_size: int = 2048
    chunk_overlap: int = 50

    # Context assembly
    max_context_tokens: int = 2000
    max_history_tokens: int = 500
    max_response_tokens: int = 200

    # Cache
    cache_ttl_seconds: int = 3600

    # RRF
    rrf_k: int = 60


settings = Settings()
