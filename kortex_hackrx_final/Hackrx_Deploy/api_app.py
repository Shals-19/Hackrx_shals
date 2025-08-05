# --- 1. Imports ---
from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import json
import os
import re
import requests
import hashlib
import time
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
API_TOKEN = os.getenv('API_TOKEN', '49092b2a30dc77e80c88e0550254ddd7928dea77103e0f05ad669ba81de92b04')
PORT = int(os.getenv('PORT', 8000))
HOST = os.getenv('HOST', '0.0.0.0')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# --- 2. Pydantic Schemas ---
class HackRxRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]
    status: str = "processing"

# --- 3. Configuration ---
CACHE_DIR = Path("./temp_indexes")
CACHE_DIR.mkdir(exist_ok=True)

# --- 4. FastAPI Application ---
app = FastAPI(title="HackRx Document Query-Retrieval System")

# Global variables for lazy loading
processor = None
loading_in_progress = False

async def initialize_processor():
    global processor, loading_in_progress

    if processor is not None or loading_in_progress:
        return

    loading_in_progress = True

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import pypdf
        import docx
        from email.parser import BytesParser
        from email.policy import default
        from io import BytesIO
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import google.generativeai as genai
        import pickle

        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDXuvJKPzcXAcEhIsJi2M-kT7mfc8Q9MYQ')
        GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash')

        genai.configure(api_key=GEMINI_API_KEY)

        class DocumentProcessor:
            @staticmethod
            def download_from_blob(url: str) -> bytes:
                response = requests.get(url)
                response.raise_for_status()
                return response.content

            @staticmethod
            def detect_document_type(path_or_url: str) -> str:
                if path_or_url.lower().endswith('.pdf'):
                    return "pdf"
                elif path_or_url.lower().endswith(('.docx', '.doc')):
                    return "docx"
                elif path_or_url.lower().endswith(('.eml', '.msg')):
                    return "email"
                return "pdf"

            @staticmethod
            def process_pdf(content: bytes) -> List[Dict]:
                reader = pypdf.PdfReader(BytesIO(content))
                return [{"content": page.extract_text() or "", "metadata": {"page": i + 1}} for i, page in enumerate(reader.pages) if page.extract_text()]

            @staticmethod
            def process_docx(content: bytes) -> List[Dict]:
                doc = docx.Document(BytesIO(content))
                text = "\n".join(para.text for para in doc.paragraphs)
                return [{"content": text, "metadata": {"page": 1}}]

            @staticmethod
            def process_email(content: bytes) -> List[Dict]:
                parser = BytesParser(policy=default)
                msg = parser.parsebytes(content)
                header = f"From: {msg['from']}\nTo: {msg['to']}\nSubject: {msg['subject']}\n\n"
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body += part.get_content()
                else:
                    body = msg.get_content()
                return [{"content": header + body, "metadata": {"page": 1}}]

            @classmethod
            def load_and_process_document(cls, url: str) -> List[Dict]:
                if url.startswith(('http://', 'https://')):
                    content = cls.download_from_blob(url)
                    name = url.split('/')[-1].split('?')[0]
                else:
                    file = Path(url)
                    if not file.exists():
                        raise FileNotFoundError(f"No file at {url}")
                    with open(file, "rb") as f:
                        content = f.read()
                    name = file.name

                doc_type = cls.detect_document_type(url)
                if doc_type == "pdf":
                    docs = cls.process_pdf(content)
                elif doc_type == "docx":
                    docs = cls.process_docx(content)
                elif doc_type == "email":
                    docs = cls.process_email(content)
                else:
                    raise ValueError("Unsupported document type")

                for d in docs:
                    d["metadata"]["source"] = name
                return docs

        class DynamicRAGProcessor:
            def __init__(self):
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                try:
                    self.llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
                except Exception as e:
                    self.llm = None

            def _split_documents(self, docs):
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = []
                for doc in docs:
                    for i, chunk in enumerate(splitter.split_text(doc['content'])):
                        chunks.append({
                            "content": chunk,
                            "metadata": {**doc['metadata'], "chunk_id": i}
                        })
                return chunks

            def _get_or_create_faiss_index(self, doc_path, docs):
                import pickle
                doc_hash = hashlib.md5(doc_path.encode()).hexdigest()
                index_file = CACHE_DIR / f"{doc_hash}.faiss"
                meta_file = CACHE_DIR / f"{doc_hash}.pkl"

                if index_file.exists() and meta_file.exists():
                    index = faiss.read_index(str(index_file))
                    with open(meta_file, 'rb') as f:
                        metadata = pickle.load(f)
                    return index, metadata

                chunks = self._split_documents(docs)
                embeddings = self.embedding_model.encode([c['content'] for c in chunks])
                index = faiss.IndexFlatL2(self.embedding_dim)
                index.add(np.array(embeddings, dtype=np.float32))
                faiss.write_index(index, str(index_file))
                with open(meta_file, 'wb') as f:
                    pickle.dump(chunks, f)
                return index, chunks

            def _llm_evaluate(self, query, context):
                if not self.llm:
                    return "Model not available."
                context_str = "\n".join([f"- {c['content']}" for c in context])
                prompt = f"""
                Answer the question using the below context. If no answer is found, say so.
                Context:
                {context_str}
                Question: {query}
                """
                response = self.llm.generate_content(prompt)
                return response.text.strip()

            def process_request(self, request: HackRxRequest):
                try:
                    docs = DocumentProcessor.load_and_process_document(request.documents)
                    index, metadata = self._get_or_create_faiss_index(request.documents, docs)
                    answers = []
                    for q in request.questions:
                        emb = self.embedding_model.encode([q])
                        top_k = min(5, len(metadata))
                        _, idxs = index.search(np.array(emb, dtype=np.float32), top_k)
                        context = [metadata[i] for i in idxs[0] if i < len(metadata)]
                        answers.append(self._llm_evaluate(q, context))
                    return {"answers": answers, "status": "complete"}
                except Exception as e:
                    return {"answers": [str(e)], "status": "error"}

        processor = DynamicRAGProcessor()

    except Exception as e:
        print(f"Initialization failed: {e}")
    finally:
        loading_in_progress = False

# --- 5. Authorization Middleware ---
def verify_api_key(authorization: str = Header(...)):
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer' or token != API_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return token
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

# --- 6. Endpoints ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: HackRxRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    global processor
    if processor is None:
        if not loading_in_progress:
            background_tasks.add_task(initialize_processor)
        return HackRxResponse(answers=["Model loading... Try again in a minute."], status="initializing")
    result = processor.process_request(request)
    return HackRxResponse(**result)

@app.get("/")
async def root(background_tasks: BackgroundTasks):
    global processor
    if processor is None and not loading_in_progress:
        background_tasks.add_task(initialize_processor)
    return {
        "status": "API is running",
        "endpoints": { "query": "/hackrx/run" },
        "version": "1.0.0",
        "model_status": "loaded" if processor else "initializing" if loading_in_progress else "not_loaded"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": "loaded" if processor else "initializing" if loading_in_progress else "not_loaded"
    }

@app.on_event("startup")
async def on_startup():
    print("Startup: processor will be loaded lazily.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_app:app", host=HOST, port=PORT, reload=DEBUG)
