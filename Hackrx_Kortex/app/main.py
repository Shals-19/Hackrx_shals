import os
from fastapi import FastAPI, Depends, HTTPException, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import router as api_router
from app.utils.config_validator import validate_on_startup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Create required directories
os.makedirs(get_settings().CACHE_DIR, exist_ok=True)
os.makedirs(get_settings().DOCUMENT_DIR, exist_ok=True)

# Validate configuration
validate_on_startup()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Kortex: LLM-Powered Query-Retrieval System",
    description="An intelligent document query system for insurance, legal, HR, and compliance domains",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to Kortex Query-Retrieval API. Use /docs for API documentation."}

# Check if .env file exists and is readable
env_path = ".env"
if os.path.exists(env_path):
    print(f"✅ .env file found at {os.path.abspath(env_path)}")
    # Try to read it to check permissions
    try:
        with open(env_path, 'r') as f:
            first_line = f.readline().strip()
        print(f"✅ .env file is readable")
    except Exception as e:
        print(f"❌ .env file exists but cannot be read: {str(e)}")
else:
    print(f"❌ .env file not found at {os.path.abspath(env_path)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)