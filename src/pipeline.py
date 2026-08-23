from typing import List, Dict, Any, Optional
from src.ingestion import ingest_directory
from src.retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.generator import GroundedGenerator

class RAGPipeline:
    def __init__(self, data_dir: str = "data"):
        print("[Pipeline] Ingesting documents...")
        self.chunks = ingest_directory(data_dir)
        
        print("[Pipeline] Building Hybrid Retriever (ChromaDB + BM25)...")
        self.retriever = HybridRetriever(self.chunks)
        
        print("[Pipeline] Initializing Reranker...")
        self.reranker = CrossEncoderReranker()
        
        print("[Pipeline] Initializing Grounded Generator...")
        self.generator = GroundedGenerator()
        print("[Pipeline] System ready for queries!")

    def query(self, query: str, top_k: int = 5, top_n: int = 2, **kwargs) -> Dict[str, Any]:
        """Runs the complete hybrid retrieval, reranking, and generation pipeline."""
        query_text = query or kwargs.get("question", "")
        
        # 1. Hybrid Retrieval (BM25 + Dense ChromaDB)
        candidates = self.retriever.retrieve(query_text, top_k=top_k)
        
        # 2. Cross-Encoder Reranking
        reranked_results = self.reranker.rerank(query_text, candidates, top_n=top_n)
        
        # 3. Format context and citations
        selected_chunks = [item["chunk"] for item in reranked_results]
        citations = [
            {
                "source": item["chunk"].get("source", "unknown"),
                "chunk_id": item["chunk"].get("chunk_id", "unknown"),
                "score": round(float(item["score"]), 4) if "score" in item else None
            }
            for item in reranked_results
        ]
        
        # 4. Grounded LLM Generation
        answer = self.generator.generate_answer(query_text, selected_chunks)
        
        return {
            "query": query_text,
            "answer": answer,
            "citations": citations,
            "retrieved_context": [c.get("text", "") for c in selected_chunks]
        }