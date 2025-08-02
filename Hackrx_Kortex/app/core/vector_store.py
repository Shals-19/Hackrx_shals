import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Set up logger
logger = logging.getLogger(__name__)

class InMemoryVectorStore:
    """Simple in-memory vector store to use as fallback when Pinecone is unavailable"""
    
    def __init__(self, max_vectors: int = 10000):
        self.vectors = {}  # id -> vector
        self.metadata = {}  # id -> metadata
        self.max_vectors = max_vectors
        
    def upsert(self, vectors: List[Dict[str, Any]]):
        for item in vectors:
            # If we're at capacity, remove oldest vectors (simple FIFO)
            if len(self.vectors) >= self.max_vectors:
                # Remove oldest 10% of vectors
                remove_count = max(1, self.max_vectors // 10)
                keys_to_remove = list(self.vectors.keys())[:remove_count]
                for key in keys_to_remove:
                    del self.vectors[key]
                    if key in self.metadata:
                        del self.metadata[key]
                        
            self.vectors[item['id']] = item['values']
            if 'metadata' in item:
                self.metadata[item['id']] = item['metadata']
                
    def query(self, vector: List[float], top_k: int = 5, include_metadata: bool = True, filter_dict: Dict = None):
        if not self.vectors:
            return {'matches': []}
            
        # Calculate cosine similarity with all stored vectors
        scores = {}
        for id, stored_vector in self.vectors.items():
            # Apply filter if provided
            if filter_dict and id in self.metadata:
                # Check if this document matches the filter
                metadata = self.metadata[id]
                should_include = True
                
                # Simple filter implementation - check for exact matches
                for filter_key, filter_value in filter_dict.items():
                    if filter_key in metadata:
                        # Handle special $eq operator
                        if isinstance(filter_value, dict) and "$eq" in filter_value:
                            if metadata[filter_key] != filter_value["$eq"]:
                                should_include = False
                                break
                        # Direct comparison
                        elif metadata[filter_key] != filter_value:
                            should_include = False
                            break
                
                # Skip this vector if it doesn't match the filter
                if not should_include:
                    continue
            
            # Calculate similarity if passes filter
            dot_product = np.dot(vector, stored_vector)
            norm_a = np.linalg.norm(vector)
            norm_b = np.linalg.norm(stored_vector)
            similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0
            scores[id] = similarity
            
        # Sort by score and get top_k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results to match Pinecone's response structure
        matches = []
        for id, score in sorted_items:
            match = {
                'id': id,
                'score': float(score)
            }
            if include_metadata and id in self.metadata:
                match['metadata'] = self.metadata[id]
            matches.append(match)
            
        return {'matches': matches}

class PineconeVectorStore:
    def __init__(self):
        self.use_fallback = False
        self.index = None
        self.fallback = InMemoryVectorStore()
        self._initialize()
        
    def _initialize(self):
        """Initialize the vector store, falling back to in-memory if Pinecone fails"""
        try:
            # Check if Pinecone is available
            if not PINECONE_AVAILABLE:
                logger.warning("Pinecone client not available, using in-memory fallback")
                self.use_fallback = True
                return
                
            # Try to initialize Pinecone
            pinecone.init(
                api_key=os.environ.get("PINECONE_API_KEY", ""),
                environment=os.environ.get("PINECONE_ENVIRONMENT", "")
            )
            
            # Check if index already exists
            index_name = os.environ.get("PINECONE_INDEX", "")
            
            if not index_name:
                logger.warning("PINECONE_INDEX not configured, using in-memory fallback")
                self.use_fallback = True
                return
            
            # Check if the index exists in the list of indexes
            existing_indexes = pinecone.list_indexes()
            
            if index_name in existing_indexes:
                logger.info(f"Connecting to existing index: {index_name}")
                self.index = pinecone.Index(index_name)
            else:
                logger.warning("Cannot create Pinecone index due to pod limitations")
                logger.warning("Using in-memory vector store as fallback")
                self.use_fallback = True
                
        except Exception as e:
            logger.error(f"Error with Pinecone, using fallback: {e}")
            self.use_fallback = True

    def upsert(self, vectors: List[Dict[str, Any]]):
        """Upsert vectors to the vector store"""
        if self.use_fallback:
            return self.fallback.upsert(vectors)
        else:
            return self.index.upsert(vectors=vectors)
            
    def query(self, vector: List[float], top_k: int = 5, include_metadata: bool = True):
        """Query the vector store"""
        if self.use_fallback:
            return self.fallback.query(vector, top_k, include_metadata)
        else:
            return self.index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)