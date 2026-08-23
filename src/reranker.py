import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class Reranker:
    def __init__(self):
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.use_cohere = bool(self.cohere_api_key and not self.cohere_api_key.startswith("your_"))
        
        if self.use_cohere:
            import cohere
            self.co_client = cohere.Client(self.cohere_api_key)
            print("[Reranker] Using Cohere Rerank API.")
        else:
            from sentence_transformers import CrossEncoder
            print("[Reranker] Using local Cross-Encoder model...")
            self.local_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, chunks: List[Dict], top_n: int = 3) -> List[Dict]:
        """Reranks retrieved candidate chunks to find the most relevant ones."""
        if not chunks:
            return []
            
        if self.use_cohere:
            docs = [c["text"] for c in chunks]
            response = self.co_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs,
                top_n=top_n
            )
            reranked = []
            for item in response.results:
                chunk = dict(chunks[item.index])
                chunk["rerank_score"] = round(item.relevance_score, 4)
                reranked.append(chunk)
            return reranked
        else:
            pairs = [[query, c["text"]] for c in chunks]
            scores = self.local_model.predict(pairs)
            scored = []
            for chunk, score in zip(chunks, scores):
                c = dict(chunk)
                c["rerank_score"] = round(float(score), 4)
                scored.append(c)
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored[:top_n]