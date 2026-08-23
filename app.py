from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
from src.pipeline import RAGPipeline

app = FastAPI(
    title="Production RAG System API",
    description="Enterprise RAG API featuring Hybrid Retrieval (BM25 + Dense Vector), Cross-Encoder Reranking, and Grounded Gemini Generation.",
    version="1.0.0"
)

# Initialize pipeline once on server startup
print("[API] Starting RAG Server...")
pipeline = RAGPipeline(data_dir="data")

# --- Request & Response Models ---
class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the encryption standard and key rotation policy?")
    retrieve_top_k: Optional[int] = Field(default=5, ge=1, le=20)
    rerank_top_n: Optional[int] = Field(default=2, ge=1, le=10)

class Citation(BaseModel):
    source: str
    chunk_id: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieved_context: Optional[List[str]] = None

# --- API Endpoints ---
@app.get("/health", tags=["Health"])
def health_check():
    """Returns the operational status of the service."""
    return {
        "status": "healthy",
        "total_chunks_indexed": len(pipeline.chunks),
        "generator_model": pipeline.generator.model_name
    }

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(request: QueryRequest):
    """Executes end-to-end hybrid retrieval, reranking, and answer generation."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        result = pipeline.query(
            question=request.question,
            retrieve_top_k=request.retrieve_top_k,
            rerank_top_n=request.rerank_top_n
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)