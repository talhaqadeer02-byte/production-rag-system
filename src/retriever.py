from typing import List, Dict
import chromadb
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """
    Hybrid Retriever combining Sparse Keyword Search (BM25) and
    Dense Vector Search (ChromaDB) using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("Retriever initialized with empty chunks list.")
            
        self.chunks = chunks
        self.chunk_map = {c["chunk_id"]: c for c in chunks}
        
        # 1. BM25 Sparse Index
        self.tokenized_corpus = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 2. ChromaDB Dense Vector Index (ephemeral in-memory)
        self.client = chromadb.Client()
        # Reset collection if exists
        try:
            self.client.delete_collection("rag_corpus")
        except Exception:
            pass
            
        self.collection = self.client.create_collection("rag_corpus")
        self.collection.add(
            documents=[c["text"] for c in chunks],
            ids=[c["chunk_id"] for c in chunks],
            metadatas=[{"source": c["source"]} for c in chunks]
        )

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict]:
        """
        Retrieves top candidate chunks using Reciprocal Rank Fusion.
        RRF Score = 1 / (rrf_k + rank_bm25) + 1 / (rrf_k + rank_vector)
        """
        # --- BM25 Keyword Search ---
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)), 
            key=lambda i: bm25_scores[i], 
            reverse=True
        )[:top_k * 2]
        
        # --- Dense Vector Search ---
        vector_res = self.collection.query(
            query_texts=[query], 
            n_results=min(top_k * 2, len(self.chunks))
        )
        vector_ids = vector_res["ids"][0] if vector_res["ids"] else []
        
        # --- Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        
        # Add BM25 ranks
        for rank, idx in enumerate(bm25_ranked_indices):
            cid = self.chunks[idx]["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        # Add Vector ranks
        for rank, cid in enumerate(vector_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by final fused score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        
        results = []
        for cid in sorted_ids:
            chunk_data = dict(self.chunk_map[cid])
            chunk_data["rrf_score"] = round(rrf_scores[cid], 5)
            results.append(chunk_data)
            
        return results

if __name__ == "__main__":
    from src.ingestion import ingest_directory
    
    # Test end-to-end ingestion and retrieval
    chunks = ingest_directory("data")
    retriever = HybridRetriever(chunks)
    
    test_query = "What is the RTO and RPO for disaster recovery?"
    top_matches = retriever.retrieve(test_query, top_k=2)
    
    print(f"\nQuery: {test_query}")
    for i, match in enumerate(top_matches, 1):
        print(f"\n--- Match {i} [Score: {match['rrf_score']}] (Source: {match['source']}, ID: {match['chunk_id']}) ---")
        print(match['text'])