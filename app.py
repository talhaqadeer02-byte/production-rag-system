from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

from src.pipeline import RAGPipeline

load_dotenv()

app = FastAPI(
    title="Production Enterprise RAG API",
    description="Hybrid RAG API with BM25 + Dense Retrieval, Cross-Encoder Reranking, and Gemini Grounded Generation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    print("[API] Initializing Production RAG Pipeline...")
    pipeline = RAGPipeline("data")
    print("[API] Pipeline ready!")

class QueryRequest(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    retrieve_top_k: Optional[int] = 5
    rerank_top_n: Optional[int] = 2

class Citation(BaseModel):
    source: str
    chunk_id: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieved_context: Optional[List[str]] = None

@app.get("/")
def root():
    return {"message": "Production RAG API is operational", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "production-rag-system"}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized.")
    
    prompt = request.query or request.question
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        result = pipeline.query(
            prompt.strip(),
            top_k=request.retrieve_top_k or 5,
            top_n=request.rerank_top_n or 2
        )
        return result
    except Exception as e:
        print(f"[API Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))