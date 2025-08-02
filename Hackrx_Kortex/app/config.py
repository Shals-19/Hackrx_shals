import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Configuration
    API_KEY: str = os.getenv("API_KEY")
    
    # LLM Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "lama3.1:latest")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "hackrx-kortex")
    
    # Document Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

    # Caching
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./data/cache")
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", 1000))
    
    # Storage
    DOCUMENT_DIR: str = os.getenv("DOCUMENT_DIR", "./data/documents")
    
    # Deployment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()