import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
except ImportError:
    genai = None

class GroundedGenerator:
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[Generator Warning] GEMINI_API_KEY environment variable not found in .env!")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generates a grounded response based strictly on the retrieved context chunks."""
        if not context_chunks:
            return "I could not find any relevant information in the document corpus to answer your question."

        context_blocks = []
        for c in context_chunks:
            source = c.get("source", "unknown")
            chunk_id = c.get("chunk_id", "unknown")
            text = c.get("text", "")
            context_blocks.append(f"[Source: {source}, Chunk: {chunk_id}]\n{text}")

        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a precise enterprise technical assistant. Answer the user's question strictly using the provided context snippets below.
If the information cannot be found in the context, state clearly that the documentation does not contain that information.
Always include provenance citations like [Source: filename, Chunk: id] for all claims.

Context:
{context_str}

Question: {query}

Answer:"""

        if not self.client:
            return f"[Mock Grounded Answer]: {context_chunks[0].get('text', '')} [Source: {context_chunks[0].get('source', 'doc')}, Chunk: {context_chunks[0].get('chunk_id', '000')}]"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            return f"Error querying Gemini API: {str(e)}"

# Backward compatibility alias
Generator = GroundedGenerator