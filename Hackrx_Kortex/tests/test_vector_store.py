import pytest
from unittest.mock import patch, MagicMock
from app.core.vector_store import PineconeVectorStore

class TestVectorStore:
    @patch("pinecone.init")
    @patch("pinecone.Index")
    def setup_method(self, mock_index, mock_init):
        """Setup test environment with mocked Pinecone"""
        self.mock_index = MagicMock()
        mock_index.return_value = self.mock_index
        self.vector_store = PineconeVectorStore()
        
    async def test_store_embeddings(self):
        """Test storing embeddings"""
        # Setup mock response
        self.mock_index.upsert.return_value = {"upserted_count": 2}
        
        chunks = [
            {
                "text": "Text 1",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"doc_id": "doc1", "page": 1}
            },
            {
                "text": "Text 2",
                "embedding": [0.4, 0.5, 0.6],
                "metadata": {"doc_id": "doc1", "page": 2}
            }
        ]
        
        result = await self.vector_store.store_embeddings(chunks)
        
        assert result["upserted_count"] == 2
        self.mock_index.upsert.assert_called_once()
        
    async def test_search(self):
        """Test vector search"""
        # Setup mock response
        self.mock_index.query.return_value = {
            "matches": [
                {
                    "id": "doc1_0",
                    "score": 0.9,
                    "metadata": {
                        "text": "Text 1",
                        "page": 1
                    }
                }
            ]
        }
        
        query_vector = [0.1, 0.2, 0.3]
        result = await self.vector_store.search(query_vector, top_k=1)
        
        assert "matches" in result
        assert len(result["matches"]) == 1
        assert result["matches"][0]["metadata"]["text"] == "Text 1"
        self.mock_index.query.assert_called_once_with(
            vector=query_vector,
            top_k=1,
            include_metadata=True,
            filter=None
        )