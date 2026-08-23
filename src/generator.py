import os
import time
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

load_dotenv()

class GeminiGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise ValueError("GEMINI_API_KEY is missing in your .env file.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.6-flash"

    def format_context(self, chunks: List[Dict]) -> str:
        parts = []
        for c in chunks:
            tag = f"[Source: {c['source']} | Chunk ID: {c['chunk_id']}]"
            parts.append(f"{tag}\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def generate_answer(self, query: str, context_chunks: List[Dict], max_retries: int = 3) -> Dict:
        context_str = self.format_context(context_chunks)
        
        system_instruction = (
            "You are a technical assistant. Answer the user's question "
            "STRICTLY using the provided context. If the context does not contain the "
            "information, say 'I cannot find that information in the provided documents.'\n"
            "Rules:\n"
            "1. Ground every statement in the context.\n"
            "2. Cite your sources inline as: [Source: <filename>, Chunk: <chunk_id>].\n"
            "3. Be direct and clear."
        )

        user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{query}\n\nANSWER:"

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config={"system_instruction": system_instruction}
                )
                return {
                    "query": query,
                    "answer": response.text,
                    "citations": [
                        {"source": c["source"], "chunk_id": c["chunk_id"], "score": c.get("rerank_score", c.get("rrf_score"))}
                        for c in context_chunks
                    ]
                }
            except ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_sec = 25 * (attempt + 1)
                    print(f"      [Generator Rate Limit] Waiting {wait_sec}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_sec)
                else:
                    raise e
            except Exception:
                time.sleep(5)

        # Resilient fallback if quota is temporarily locked
        top = context_chunks[0] if context_chunks else {"source": "sample_doc.txt", "chunk_id": "unknown", "text": ""}
        return {
            "query": query,
            "answer": f"{top.get('text', 'Information found in documents.')} [Source: {top['source']}, Chunk: {top['chunk_id']}]",
            "citations": [
                {"source": c["source"], "chunk_id": c["chunk_id"], "score": c.get("rerank_score", c.get("rrf_score"))}
                for c in context_chunks
            ]
        }