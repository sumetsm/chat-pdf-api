# Chat With PDF - Multi-Agent RAG System

An intelligent question-answering system built with **LangGraph** that can handle queries over PDF documents with web search fallback capabilities, now using **Groq API** for LLM inference.

## Architecture Overview

The system implements a **multi-agent architecture** using LangGraph with the following agents:

### Agent Flow

```
Question Input → Clarification Agent → PDF Retrieval Agent → Routing Agent → Context Merger → Answer Generation Agent → Response
```

### Agent Descriptions

1. **Clarification Agent**: Analyzes incoming questions for ambiguity and provides clarification when needed
2. **PDF Retrieval Agent**: Searches the vector database for relevant document chunks using RAG
3. **Routing Agent**: Decides whether to perform web search based on PDF context availability
4. **Context Merger**: Combines PDF and web search results into unified context
5. **Answer Generation Agent**: Generates final response based on available context via **Groq LLM API**

### Key Components

* **FastAPI**: RESTful API server with session management
* **LangGraph**: Multi-agent orchestration framework
* **ChromaDB**: Vector database for document embeddings
* **HuggingFace Embeddings**: BGE-small-en-v1.5 for text embeddings
* **Groq API**: Cloud LLM for inference
* **RAG Pipeline**: Retrieval-Augmented Generation for PDF queries

---

## Features

* ✅ PDF document ingestion with chunking
* ✅ Semantic search over PDF content
* ✅ Web search fallback for out-of-scope queries
* ✅ Session-based memory management
* ✅ Query clarification for ambiguous questions
* ✅ RESTful API endpoints
* ✅ Docker containerization

---

## Quick Start

### Prerequisites

* Docker & Docker Compose
* Groq API Key (set via environment variable `GROQ_API_KEY`)
* At least 8GB RAM (optional if running large embeddings or multiple PDFs)

### Setup & Run

1. **Clone and prepare**:

```bash
git clone https://github.com/sumetsm/chat-pdf-api.git
cd chat-pdf-api
mkdir pdfs
```

2. **Add your PDF files**:

```bash
# Place your academic papers in the pdfs/ directory
cp your_papers/*.pdf pdfs/
```

3. **Set Groq API Key**:

```bash
# In .env file
GROQ_API_KEY=your_api_key_here
```

4. **Ingest PDFs**:

```bash
docker exec -it chat-pdf-api-chat-pdf-api-1 python ingest_pdfs.py
```

5. **Start services**:

```bash
docker-compose up -d --build
```

6. **Test the API**:

```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"question": "Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?", "session_id": "test"}'
```

---

## API Endpoints

### POST `/ask`

Ask questions to the system.

**Request**:

```json
{
  "question": "What is the accuracy of the model on Spider dataset?",
  "session_id": "user123"
}
```

**Response**:

```json
{
  "answer": "Based on the PDF documents...",
  "session_id": "user123",
  "context_used": "PDF, Web"
}
```

### POST `/clear-memory`

Clear session memory.

**Request**:

```json
{
  "session_id": "user123"
}
```

### GET `/health`

Health check endpoint.

---

💡 **Note:** Make sure the environment variable `GROQ_API_KEY` is set before running the service. The system will use it to call Groq API for LLM inference.

---
