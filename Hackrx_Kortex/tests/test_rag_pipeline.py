import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.core.rag_pipeline import RAGPipeline

class TestRAGPipeline:
    def setup_method(self):
        """Setup test environment"""
        # Create mocks for all dependencies
        self.mock_document_processor = AsyncMock()
        self.mock_embedding_service = AsyncMock()
        self.mock_vector_store = AsyncMock()
        self.mock_llm_service = AsyncMock()
        self.mock_cache_manager = MagicMock()
        
        # Initialize the RAG pipeline with mocks
        self.pipeline = RAGPipeline(
            document_processor=self.mock_document_processor,
            embedding_service=self.mock_embedding_service,
            vector_store=self.mock_vector_store,
            llm_service=self.mock_llm_service,
            cache_manager=self.mock_cache_manager
        )
        
    async def test_process_request_cached_document(self):
        """Test processing with cached document"""
        # Setup cache hit for document
        self.mock_cache_manager.get_document_cache.return_value = [
            {"text": "Cached chunk", "embedding": [0.1, 0.2, 0.3]}
        ]
        
        # Setup cache miss for query
        self.mock_cache_manager.get_query_cache.return_value = None
        
        # Setup embedding for question
        self.mock_embedding_service.create_embeddings.return_value = [[0.7, 0.8, 0.9]]
        
        # Setup vector search
        self.mock_vector_store.search.return_value = {
            "matches": [
                {
                    "metadata": {
                        "text": "Relevant text",
                        "page": 1
                    }
                }
            ]
        }
        
        # Setup LLM response
        self.mock_llm_service.generate_answer.return_value = "This is the answer"
        
        document_url = "http://example.com/doc.pdf"
        questions = ["What is this about?"]
        
        answers = await self.pipeline.process_request(document_url, questions)
        
        assert len(answers) == 1
        assert answers[0] == "This is the answer"
        self.mock_document_processor.process_document.assert_not_called()
        self.mock_vector_store.store_embeddings.assert_not_called()
        self.mock_llm_service.generate_answer.assert_called_once()