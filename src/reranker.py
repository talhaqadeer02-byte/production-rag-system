from typing import List, Dict, Any
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = None
        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(model_name)
            except Exception as e:
                print(f"[Reranker Warning] Model load failed ({e}), using pass-through ranking.")
                self.model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 2) -> List[Dict[str, Any]]:
        """Scores candidate chunks against the query using a Cross-Encoder."""
        if not candidates:
            return []

        if self.model is None:
            return [{"chunk": c, "score": 1.0} for c in candidates[:top_n]]

        # Prepare (query, text) pairs for scoring
        pairs = [[query, c.get("text", "")] for c in candidates]
        scores = self.model.predict(pairs)

        scored_candidates = [
            {"chunk": chunk, "score": float(score)}
            for chunk, score in zip(candidates, scores)
        ]

        # Rank descending by score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_n]

# Backward compatibility alias
Reranker = CrossEncoderReranker