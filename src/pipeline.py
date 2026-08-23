import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict
from src.ingestion import ingest_directory
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.generator import GeminiGenerator

class RAGPipeline:
    def __init__(self, data_dir: str = "data"):
        print("[Pipeline] Initializing RAG components...")
        self.chunks = ingest_directory(data_dir)
        if not self.chunks:
            raise ValueError(f"No documents found in '{data_dir}'. Add text/pdf files.")
            
        self.retriever = HybridRetriever(self.chunks)
        self.reranker = Reranker()
        self.generator = GeminiGenerator()
        print("[Pipeline] System ready for queries!\n")

    def query(self, question: str, retrieve_top_k: int = 5, rerank_top_n: int = 2) -> Dict:
        # 1. Hybrid retrieval (BM25 + Vector)
        candidates = self.retriever.retrieve(question, top_k=retrieve_top_k)
        
        # 2. Reranking
        top_chunks = self.reranker.rerank(question, candidates, top_n=rerank_top_n)
        
        # 3. Generation with citations
        result = self.generator.generate_answer(question, top_chunks)
        return result

if __name__ == "__main__":
    rag = RAGPipeline("data")
    
    question = "What encryption standard and key rotation policy are used?"
    output = rag.query(question)
    
    print("=" * 60)
    print(f"QUESTION: {output['query']}")
    print("=" * 60)
    print(f"ANSWER:\n{output['answer']}\n")
    print("SOURCES USED:")
    for c in output["citations"]:
        print(f"- File: {c['source']} | Chunk ID: {c['chunk_id']}")
    print("=" * 60)