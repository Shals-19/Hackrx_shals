from fastapi import APIRouter, Depends, HTTPException, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.core.rag_pipeline import RAGPipeline
from app.api.dependencies import get_rag_pipeline as get_pipeline_singleton

router = APIRouter()
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

async def get_rag_pipeline():
    """Get RAG pipeline instance for each request"""
    pipeline = RAGPipeline()
    await pipeline.initialize()
    return pipeline

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != get_settings().API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return True

@router.post("/hackrx/run", response_model=QueryResponse)
@limiter.limit("10/minute")
async def process_query(
    request: Request,
    query_request: QueryRequest, 
    authenticated: bool = Depends(verify_token),
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Process documents and answer questions based on their content
    
    - **documents**: URL to the document(s) to process (PDF, DOCX, email)
    - **questions**: List of questions to answer based on the document
    """
    try:
        # Convert HttpUrl to string before passing it to the pipeline
        document_url = str(query_request.documents)
        
        # Process the document and answer questions
        answers = await pipeline.process_request(document_url, query_request.questions)
        
        return QueryResponse(answers=answers)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )