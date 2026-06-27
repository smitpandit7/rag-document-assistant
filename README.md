# 🤖 AI Document Assistant

A production-ready **RAG (Retrieval-Augmented Generation)** powered document assistant that lets users upload PDF documents and ask questions about them using natural language. Built with FastAPI, ChromaDB, SentenceTransformers, and Groq (LLaMA 3.3 70B).

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Authentication](#authentication)
- [RAG Pipeline](#rag-pipeline)
- [Docker](#docker)
- [Environment Variables](#environment-variables)

---

## Overview

This project was built as part of the **IR INFOTECH AI/ML Internship — Round 2 Evaluation**.

The system allows users to:
1. Register and login securely
2. Upload one or multiple PDF documents
3. Ask natural language questions about the uploaded documents
4. Get AI-generated answers with source references (page numbers, file names)
5. View full conversation history per session
6. Stream responses token by token in real time

---

## Architecture

```
User
 │
 ▼
FastAPI Backend
 │
 ├── Auth Layer (JWT + bcrypt + SQLite)
 │
 ├── PDF Upload
 │     └── PyMuPDF → Text Extraction
 │           └── Chunker → Overlapping text chunks
 │                 └── SentenceTransformers → Embeddings (384-dim)
 │                       └── ChromaDB → Vector Storage
 │
 └── Question Answering
       └── User Question → Embed → ChromaDB Similarity Search
             └── Top-K Chunks → Prompt Builder
                   └── Groq API (LLaMA 3.3 70B) → Answer + Sources
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | Groq — LLaMA 3.3 70B |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| PDF Extraction | PyMuPDF (fitz) |
| Authentication | JWT (python-jose) + bcrypt (passlib) |
| Database | SQLite (user storage) |
| Server | Uvicorn |
| Containerization | Docker + Docker Compose |

---

## Features

### Core
- ✅ PDF Upload API (single and multi-document)
- ✅ Text Extraction (page by page using PyMuPDF)
- ✅ Document Chunking (overlapping sliding window with sentence boundary detection)
- ✅ Embedding Generation (all-MiniLM-L6-v2, 384-dim vectors)
- ✅ ChromaDB Vector Storage (per-document + shared collections)
- ✅ Question Answering with Source References (file name, page number, similarity score)
- ✅ Chat History API (per session, sliding window memory)

### Bonus
- ✅ Multi-document support (search across all uploaded PDFs simultaneously)
- ✅ Dockerization (Dockerfile + docker-compose)
- ✅ Authentication (JWT-based register/login system)
- ✅ Streaming responses (Server-Sent Events via `/chat/ask/stream`)

---

## Project Structure

```
rag-document-assistant/
│
├── app/
│   ├── main.py                  # FastAPI app, routers, startup
│   │
│   ├── routes/
│   │   ├── auth.py              # Register, Login, /me
│   │   ├── upload.py            # PDF upload, list, delete
│   │   ├── chat.py              # /ask and /ask/stream
│   │   └── history.py           # Session history endpoints
│   │
│   ├── services/
│   │   ├── pdf_loader.py        # PyMuPDF text extraction
│   │   ├── chunker.py           # Overlapping text chunking
│   │   ├── embedding.py         # SentenceTransformer embeddings
│   │   ├── vector_store.py      # ChromaDB operations
│   │   └── rag.py               # Full RAG pipeline + chat history
│   │
│   ├── core/
│   │   ├── config.py            # Environment config
│   │   ├── logger.py            # Rotating file + console logger
│   │   ├── auth.py              # JWT token logic
│   │   ├── database.py          # SQLite connection
│   │   └── user_service.py      # User CRUD + password hashing
│   │
│   └── models/
│       └── schemas.py           # Pydantic schemas
│
├── uploads/                     # Uploaded PDFs stored here
├── chroma_db/                   # ChromaDB vector data
├── logs/                        # Rotating log files
├── frontend/
│   └── ui.html                  # Simple test UI
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Groq API key — get one free at [console.groq.com](https://console.groq.com)
- Docker (optional, for containerized run)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-document-assistant.git
cd rag-document-assistant
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the server

```bash
python -m uvicorn app.main:app --reload
```

Server runs at: `http://localhost:8000`

API docs at: `http://localhost:8000/docs`

---

## API Documentation

Full interactive API docs are available at `/docs` (Swagger UI) after starting the server.

### Endpoints Summary

#### Authentication
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/auth/register` | Create new account | No |
| POST | `/auth/login` | Login, returns JWT token | No |
| GET | `/auth/me` | Get current user profile | Yes |

#### Documents
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/documents/upload` | Upload one or multiple PDFs | Yes |
| GET | `/documents/` | List all uploaded documents | Yes |
| DELETE | `/documents/{doc_id}` | Delete a document | Yes |

#### Chat
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/chat/ask` | Ask a question (full response) | Yes |
| POST | `/chat/ask/stream` | Ask a question (streaming SSE) | Yes |

#### History
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| GET | `/history/{session_id}` | Get chat history for a session | Yes |
| GET | `/history/` | List all active sessions | Yes |
| DELETE | `/history/{session_id}` | Clear session history | Yes |

#### Health
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| GET | `/` | App status | No |
| GET | `/health` | Model and system info | No |

---

## Authentication

The system uses **JWT Bearer token authentication**.

### Flow

```
1. POST /auth/register  →  { name, email, password }
2. POST /auth/login     →  returns { access_token, token_type }
3. All protected routes →  Header: Authorization: Bearer <token>
```

### In Swagger UI

1. Call `POST /auth/login`
2. Copy the `access_token` from the response
3. Click the **Authorize** button (top right)
4. Paste the token and click Authorize
5. All subsequent requests will include the token automatically

---

## RAG Pipeline

The RAG pipeline processes documents and questions in the following steps:

### Document Ingestion
```
PDF File
  → PyMuPDF extracts text page by page
  → Chunker splits into 500-char overlapping chunks (100-char overlap)
  → SentenceTransformer encodes chunks into 384-dim vectors
  → ChromaDB stores vectors + metadata (page_num, source_file, doc_id)
```

### Question Answering
```
User Question
  → SentenceTransformer encodes question into 384-dim vector
  → ChromaDB cosine similarity search → Top-5 most relevant chunks
  → Prompt builder combines: system instructions + chat history + context chunks + question
  → Groq API (LLaMA 3.3 70B) generates grounded answer
  → Response includes answer + source references (file, page, similarity score)
```

### Key Design Decisions

- **Page-level chunking** — chunks tagged with page numbers for precise source references
- **Sentence boundary detection** — chunks cut at `.!?` instead of mid-sentence
- **Per-document + shared collections** — ChromaDB stores chunks in both a per-doc collection and a shared `all_documents` collection, enabling both single-doc and multi-doc search
- **Sliding window chat history** — last 10 turns sent to LLM for context-aware follow-up questions
- **Grounded answers** — prompt explicitly instructs LLM to answer only from context, cite `[Context N]` references, and say "I don't know" if information isn't present

---

## Docker

### Run with Docker Compose

```bash
# Build and start
docker compose up --build

# Run in background
docker compose up --build -d

# Stop
docker compose down
```

### Run with Docker only

```bash
docker build -t rag-document-assistant .
docker run -p 8000:8000 --env-file .env rag-document-assistant
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Your Groq API key |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model to use |
| `API_KEY` | No | `secret-key-123` | API key for simple auth |

---

## Testing

A pipeline test script is included to verify all components work correctly:

```bash
python test_rag_pipeline.py
```

This tests all 7 steps: PDF extraction → chunking → embedding → ChromaDB storage → retrieval → Groq API → full RAG answer.

---

## Screen Recording

📹 [Link to screen recording](#) ← add your link here before submitting

---

*Built by Smit Pandit — IR INFOTECH AI/ML Internship Round 2*