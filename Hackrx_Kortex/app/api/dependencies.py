from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from app.config import get_settings
from app.core.document_processor import DocumentProcessor
from app.core.embeddings import EmbeddingService
from app.core.vector_store import PineconeVectorStore
from app.core.llm_service import OllamaService
from app.core.cache import CacheManager
from app.core.rag_pipeline import RAGPipeline

# Authentication security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """
    Verify the authentication token
    
    Args:
        credentials: Authentication credentials
        
    Returns:
        True if authenticated, raises exception otherwise
    """
    if credentials.credentials != get_settings().API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return True

# Singleton instances for services
_document_processor: Optional[DocumentProcessor] = None
_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[PineconeVectorStore] = None
_llm_service: Optional[OllamaService] = None
_cache_manager: Optional[CacheManager] = None
_rag_pipeline: Optional[RAGPipeline] = None

def get_document_processor() -> DocumentProcessor:
    """Get document processor singleton instance"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor

def get_embedding_service() -> EmbeddingService:
    """Get embedding service singleton instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

def get_vector_store() -> PineconeVectorStore:
    """Get vector store singleton instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = PineconeVectorStore()
        # Make sure the vector store is initialized
        _vector_store._initialize()
    return _vector_store

def get_llm_service() -> OllamaService:
    """Get LLM service singleton instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = OllamaService()
    return _llm_service

def get_cache_manager() -> CacheManager:
    """Get cache manager singleton instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def get_rag_pipeline() -> RAGPipeline:
    """Get RAG pipeline singleton instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            document_processor=get_document_processor(),
            embedding_service=get_embedding_service(),
            vector_store=get_vector_store(),
            llm_service=get_llm_service(),
            cache_manager=get_cache_manager()
        )
    return _rag_pipeline