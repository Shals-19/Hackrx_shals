from typing import List, Dict, Any
import asyncio
import time

from app.core.document_processor import DocumentProcessor
from app.core.embeddings import EmbeddingService
from app.core.vector_store import PineconeVectorStore
from app.core.llm_service import OllamaService
from app.core.cache import CacheManager
from app.utils.response_formatter import clean_response

class RAGPipeline:
    def __init__(
        self, 
        document_processor: DocumentProcessor,
        embedding_service: EmbeddingService,
        vector_store: PineconeVectorStore,
        llm_service: OllamaService,
        cache_manager: CacheManager
    ):
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.cache_manager = cache_manager
    
    async def process_request(self, document_url: str, questions: List[str]) -> List[str]:
        """
        Process a complete query request
        
        Args:
            document_url: URL of the document to process
            questions: List of questions to answer
            
        Returns:
            List of answers to the questions
        """
        # Process document (with caching)
        await self._process_document(document_url)
        
        # Answer all questions
        answers = []
        for question in questions:
            # Check query cache first
            cached_answer = self.cache_manager.get_query_cache(document_url, question)
            if cached_answer:
                answers.append(cached_answer)
                continue
            
            # Generate answer for new question
            answer = await self._answer_question(document_url, question)
            
            # Cache the query result
            self.cache_manager.cache_query(document_url, question, answer)
            
            answers.append(answer)
        
        return answers
    
    async def _process_document(self, document_url: str) -> None:
        """Process a document and store its embeddings"""
        # Check document cache
        cached_chunks = self.cache_manager.get_document_cache(document_url)
        if cached_chunks:
            print(f"Using cached document: {document_url}")
            return
        
        # Process new document
        print(f"Processing document: {document_url}")
        
        # Get document chunks
        chunks = await self.document_processor.process_document(document_url)
        
        # Create embeddings for chunks
        chunks_with_embeddings = await self.embedding_service.process_document_chunks(chunks)
        
        # Store in vector database
        await self.vector_store.store_embeddings(chunks_with_embeddings)
        
        # Cache processed document
        self.cache_manager.cache_document(document_url, chunks_with_embeddings)
    
    async def _answer_question(self, document_url: str, question: str) -> str:
        """Answer a single question about a document"""
        # Get embedding for question
        question_embedding = await self.embedding_service.create_embeddings([question])
        
        # Create filter for specific document
        doc_id = self._generate_doc_id_from_url(document_url)
        filter_dict = {"doc_id": {"$eq": doc_id}}
        
        # Retrieve relevant chunks
        search_results = await self.vector_store.search(
            question_embedding[0], 
            top_k=5,
            filter_dict=filter_dict
        )
        
        # Format context from chunks
        context = self._format_context(search_results)
        
        # Generate answer with LLM
        answer = await self.llm_service.generate_answer(question, context)
        
        # Clean and format the response
        cleaned_answer = clean_response(answer)
        
        return cleaned_answer
    
    def _format_context(self, search_results: Dict) -> str:
        """Format retrieved chunks into context for the LLM"""
        context = ""
        for match in search_results["matches"]:
            metadata = match["metadata"]
            text = metadata.get("text", "")
            page = metadata.get("page", "N/A")
            
            context += f"[Page {page}]: {text}\n\n"
        
        return context.strip()
    
    def _generate_doc_id_from_url(self, url: str) -> str:
        """Generate document ID from URL (same as in DocumentProcessor)"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
    
    async def process(self, request):
        """Process the request coming from the API endpoint"""
        document_url = str(request.documents)
        return {"answers": await self.process_request(document_url, request.questions)}