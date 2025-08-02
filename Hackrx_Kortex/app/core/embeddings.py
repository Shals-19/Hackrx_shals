import openai
import os
from typing import List, Dict, Any
import json
import hashlib
from pathlib import Path
import asyncio

from app.config import get_settings

class EmbeddingService:
    def __init__(self):
        self.settings = get_settings()
        openai.api_key = self.settings.OPENAI_API_KEY
        self.model = self.settings.EMBEDDING_MODEL
        self.cache_dir = Path(self.settings.CACHE_DIR) / "embeddings"
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with retry mechanism and caching"""
        if not texts:
            return []
            
        # Check cache first
        uncached_texts = []
        uncached_indices = []
        
        # Create results array of the right size
        results = [None] * len(texts)
        
        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            embedding = self._get_cached_embedding(text)
            if embedding:
                results[i] = embedding
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If there are uncached texts, get embeddings from API with retries
        if uncached_texts:
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    print(f"🧠 Generating embeddings for {len(uncached_texts)} texts (attempt {attempt+1})")
                    response = await openai.Embedding.acreate(
                        input=uncached_texts,
                        model=self.model
                    )
                    
                    # Process new embeddings
                    new_embeddings = [item["embedding"] for item in response["data"]]
                    
                    # Cache new embeddings and add to results
                    for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        self._cache_embedding(text, embedding)
                        results[uncached_indices[idx]] = embedding
                    
                    print(f"✅ Embeddings generated successfully")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Embedding generation failed: {str(e)}. Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"❌ Embedding generation failed after {max_retries} attempts: {str(e)}")
                        raise
        
        # Ensure all embeddings were generated
        if None in results:
            raise ValueError("Failed to generate all embeddings")
            
        return results
    
    async def process_document_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process document chunks to create embeddings
        
        Args:
            chunks: List of document chunks with text and metadata
            
        Returns:
            List of chunks with embeddings added
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.create_embeddings(texts)
        
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
            
        return chunks
    
    def _get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding from cache if available"""
        cache_key = self._compute_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for future use"""
        cache_key = self._compute_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)
    
    def _compute_cache_key(self, text: str) -> str:
        """Compute a cache key for a text string"""
        return hashlib.md5(f"{text}-{self.model}".encode()).hexdigest()