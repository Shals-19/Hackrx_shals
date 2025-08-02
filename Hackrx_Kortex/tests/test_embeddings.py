import pytest
from unittest.mock import patch, MagicMock
from app.core.embeddings import EmbeddingService

class TestEmbeddingService:
    def setup_method(self):
        """Setup test environment"""
        self.embedding_service = EmbeddingService()
        
    @patch("openai.Embedding.acreate")
    async def test_create_embeddings(self, mock_acreate):
        """Test embedding creation"""
        # Mock OpenAI response
        mock_acreate.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }
        
        texts = ["Text 1", "Text 2"]
        result = await self.embedding_service.create_embeddings(texts)
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        
    @patch.object(EmbeddingService, "create_embeddings")
    async def test_process_document_chunks(self, mock_create_embeddings):
        """Test document chunk processing"""
        mock_create_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        chunks = [
            {"text": "Text 1", "metadata": {"page": 1}},
            {"text": "Text 2", "metadata": {"page": 2}}
        ]
        
        result = await self.embedding_service.process_document_chunks(chunks)
        
        assert len(result) == 2
        assert "embedding" in result[0]
        assert "embedding" in result[1]
        assert result[0]["embedding"] == [0.1, 0.2, 0.3]