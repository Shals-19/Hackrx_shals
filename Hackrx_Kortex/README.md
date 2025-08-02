# Kortex: LLM-Powered Query-Retrieval System

An intelligent document query-retrieval system that can process large documents and answer natural language queries for insurance, legal, HR, and compliance domains.

## Features

- Processes PDF, DOCX, and email documents
- Semantic search using embeddings and Pinecone vector database
- Context-aware answers using Llama LLM via Ollama
- Document and embedding caching for performance
- Structured JSON responses
- RESTful API built with FastAPI

## Architecture

![Architecture Diagram](docs/architecture.png)

The system follows a 6-component workflow:
1. **Input Documents**: Accepts PDF/DOCX/Email documents via URL
2. **Document Processing**: Extracts and chunks text using semantic chunking
3. **Embedding Search**: Uses OpenAI embeddings with Pinecone vector database
4. **Clause Matching**: Semantically matches queries to relevant document sections
5. **Logic Evaluation**: Processes decisions based on matched content
6. **JSON Response**: Generates structured, accurate answers

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) with Llama model installed
- Pinecone account (free tier available)
- OpenAI API key

### Installation

1. Clone the repository: