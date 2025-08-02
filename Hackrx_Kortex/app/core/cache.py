import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

from app.config import get_settings

class CacheManager:
    def __init__(self):
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.CACHE_DIR)
        
        # Create cache directories
        self.document_cache_dir = self.cache_dir / "documents"
        self.query_cache_dir = self.cache_dir / "queries"
        
        os.makedirs(self.document_cache_dir, exist_ok=True)
        os.makedirs(self.query_cache_dir, exist_ok=True)
        
        self.max_cache_size = self.settings.MAX_CACHE_SIZE
    
    def get_document_cache(self, document_url: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get document chunks from cache if available
        
        Args:
            document_url: URL of the document
            
        Returns:
            List of document chunks if cached, None otherwise
        """
        cache_key = self._compute_cache_key(document_url)
        cache_file = self.document_cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Check if cache is still valid (1 day)
                if time.time() - data.get("timestamp", 0) < 86400:
                    return data.get("chunks")
            except:
                pass
        
        return None
    
    def cache_document(self, document_url: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Cache document chunks for future use
        
        Args:
            document_url: URL of the document
            chunks: List of document chunks
        """
        cache_key = self._compute_cache_key(document_url)
        cache_file = self.document_cache_dir / f"{cache_key}.json"
        
        # Store with timestamp for cache expiration
        data = {
            "timestamp": time.time(),
            "chunks": chunks
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
            
        self._manage_cache_size(self.document_cache_dir)
    
    def get_query_cache(self, document_url: str, question: str) -> Optional[str]:
        """
        Get query result from cache if available
        
        Args:
            document_url: URL of the document
            question: Question text
            
        Returns:
            Cached answer if available, None otherwise
        """
        cache_key = self._compute_cache_key(f"{document_url}-{question}")
        cache_file = self.query_cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Check if cache is still valid (1 hour)
                if time.time() - data.get("timestamp", 0) < 3600:
                    return data.get("answer")
            except:
                pass
        
        return None
    
    def cache_query(self, document_url: str, question: str, answer: str) -> None:
        """
        Cache query result
        
        Args:
            document_url: URL of the document
            question: Question text
            answer: Generated answer
        """
        cache_key = self._compute_cache_key(f"{document_url}-{question}")
        cache_file = self.query_cache_dir / f"{cache_key}.json"
        
        # Store with timestamp for cache expiration
        data = {
            "timestamp": time.time(),
            "answer": answer
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
            
        self._manage_cache_size(self.query_cache_dir)
    
    def _compute_cache_key(self, text: str) -> str:
        """Compute a cache key for a text string"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _manage_cache_size(self, cache_dir: Path) -> None:
        """Remove oldest files if cache exceeds max size"""
        files = list(cache_dir.glob("*.json"))
        if len(files) > self.max_cache_size:
            # Sort files by modification time
            files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest files until we're back under max cache size
            for file in files[:len(files) - self.max_cache_size]:
                try:
                    file.unlink()
                except:
                    pass