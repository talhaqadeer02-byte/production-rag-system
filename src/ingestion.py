import os
import uuid
from typing import List, Dict
from pypdf import PdfReader

def load_document(file_path: str) -> str:
    """Extract text from .txt or .pdf files."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
    else:
        print(f"Skipping unsupported file type: {file_path}")
        
    return text

def semantic_chunk_text(text: str, source_name: str, max_chunk_chars: int = 600) -> List[Dict]:
    """
    Splits text by paragraph boundaries while respecting semantic context
    instead of arbitrary fixed token slices.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        cleaned_para = para.strip()
        if not cleaned_para:
            continue
            
        # Group paragraphs together until max_chunk_chars threshold
        if len(current_chunk) + len(cleaned_para) <= max_chunk_chars:
            current_chunk += ("\n\n" + cleaned_para if current_chunk else cleaned_para)
        else:
            if current_chunk:
                chunks.append({
                    "chunk_id": str(uuid.uuid4())[:8],
                    "text": current_chunk,
                    "source": source_name
                })
            current_chunk = cleaned_para

    if current_chunk:
        chunks.append({
            "chunk_id": str(uuid.uuid4())[:8],
            "text": current_chunk,
            "source": source_name
        })

    return chunks

def ingest_directory(data_dir: str = "data") -> List[Dict]:
    """Loads and chunks all documents from the data directory."""
    all_chunks = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return all_chunks

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            text = load_document(file_path)
            if text.strip():
                chunks = semantic_chunk_text(text, source_name=filename)
                all_chunks.extend(chunks)
                print(f"Loaded {filename}: {len(chunks)} chunks created.")
                
    return all_chunks

if __name__ == "__main__":
    # Quick standalone test
    chunks = ingest_directory("data")
    print(f"Total chunks ingested: {len(chunks)}")