"""
Configuration Validation Module

This module validates all configuration values at startup to ensure the application
can function properly and securely before handling any requests.
"""
import os
import sys
from typing import Dict, Any, List, Optional

from app.config import get_settings

def validate_api_key() -> Optional[str]:
    """Validate that API_KEY is set and secure"""
    settings = get_settings()
    
    # Check API key is set
    if not settings.API_KEY:
        return "API_KEY environment variable must be set"
        
    # Check API key meets minimum security requirements
    if len(settings.API_KEY) < 32:
        return "API_KEY must be at least 32 characters long"
        
    return None

def validate_document_directories() -> Optional[str]:
    """Validate document and cache directories"""
    settings = get_settings()
    
    # Check cache directory
    try:
        os.makedirs(settings.CACHE_DIR, exist_ok=True)
        test_file_path = os.path.join(settings.CACHE_DIR, "write_test.txt")
        with open(test_file_path, 'w') as f:
            f.write("test")
        os.remove(test_file_path)
    except Exception as e:
        return f"Cache directory ({settings.CACHE_DIR}) is not writable: {str(e)}"
        
    # Check document directory
    try:
        os.makedirs(settings.DOCUMENT_DIR, exist_ok=True)
        test_file_path = os.path.join(settings.DOCUMENT_DIR, "write_test.txt")
        with open(test_file_path, 'w') as f:
            f.write("test")
        os.remove(test_file_path)
    except Exception as e:
        return f"Document directory ({settings.DOCUMENT_DIR}) is not writable: {str(e)}"
        
    return None

def validate_ollama_config() -> Optional[str]:
    """Validate Ollama configuration if using local models"""
    settings = get_settings()
    
    # Skip validation if we're not using Ollama (i.e., if using OpenAI exclusively)
    if not settings.OLLAMA_BASE_URL or settings.OLLAMA_BASE_URL.startswith("http://none"):
        return None
        
    import requests
    
    try:
        # Test connection to Ollama
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            return f"Cannot connect to Ollama server: HTTP {response.status_code}"
            
        # Check if the specified model exists
        models = response.json().get("models", [])
        model_names = [model.get("name") for model in models]
        if settings.LLM_MODEL not in model_names:
            return f"Model {settings.LLM_MODEL} not found in Ollama. Available models: {', '.join(model_names)}"
    except requests.RequestException as e:
        return f"Cannot connect to Ollama server: {str(e)}"
        
    return None

def validate_openai_config() -> Optional[str]:
    """Validate OpenAI configuration"""
    settings = get_settings()
    
    # Skip validation if we're not using OpenAI
    if not settings.OPENAI_API_KEY:
        return None
        
    if len(settings.OPENAI_API_KEY) < 20:
        return "OPENAI_API_KEY seems invalid (too short)"
        
    return None

def validate_pinecone_config() -> Optional[str]:
    """Validate Pinecone configuration"""
    settings = get_settings()
    
    # Skip validation if we're not using Pinecone
    if not settings.PINECONE_API_KEY:
        return None
        
    if len(settings.PINECONE_API_KEY) < 20:
        return "PINECONE_API_KEY seems invalid (too short)"
        
    if not settings.PINECONE_INDEX:
        return "PINECONE_INDEX must be set when using Pinecone"
        
    if not settings.PINECONE_ENVIRONMENT:
        return "PINECONE_ENVIRONMENT must be set when using Pinecone"
        
    return None

def validate_all() -> List[str]:
    """Run all validation checks and return any errors"""
    validators = [
        validate_api_key,
        validate_document_directories,
        validate_ollama_config,
        validate_openai_config,
        validate_pinecone_config
    ]
    
    errors = []
    for validator in validators:
        try:
            error = validator()
            if error:
                errors.append(error)
        except Exception as e:
            errors.append(f"Error in {validator.__name__}: {str(e)}")
    
    return errors

def validate_on_startup() -> None:
    """Validate all configurations on startup and exit if invalid"""
    errors = validate_all()
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        
        # Exit with error code in development environment
        # In production, we may want to continue with warnings
        if get_settings().ENVIRONMENT == "development":
            sys.exit(1)
