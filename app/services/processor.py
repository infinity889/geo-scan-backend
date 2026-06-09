from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from app.services.model_tasks import embed_texts
from app.services.vector_store import vector_store

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

async def process_document_async(
    document_id: str,
    document_name: str,
    file_path: Path,
    log_callback: Callable[[str], None]
) -> None:
    log_callback("Starting document processing...")
    
    if file_path.suffix.lower() == ".pdf":
        log_callback("Opening PDF file...")
        try:
            reader = PdfReader(file_path)
        except Exception as e:
            log_callback(f"Failed to open PDF: {e}")
            return

        all_chunks = []
        all_pages = []
        
        log_callback("Extracting and chunking text...")
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                page_chunks = chunk_text(text)
                all_chunks.extend(page_chunks)
                all_pages.extend([str(page_num + 1)] * len(page_chunks))
                
        if not all_chunks:
            log_callback("No text found in document.")
            return

        log_callback(f"Generated {len(all_chunks)} chunks. Generating embeddings...")
        
        batch_size = 10
        all_embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch_texts = all_chunks[i : i + batch_size]
            try:
                response = await embed_texts(batch_texts)
                all_embeddings.extend(response.embeddings)
                log_callback(f"Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks...")
            except Exception as e:
                log_callback(f"Embedding failed at batch {i}: {e}")
                # Fallback to zeros just so we don't crash the whole pipeline
                dimension = 1536  # typical
                if response and hasattr(response, "dimensions") and response.dimensions:
                    dimension = response.dimensions
                elif all_embeddings:
                    dimension = len(all_embeddings[0])
                all_embeddings.extend([[0.0] * dimension for _ in batch_texts])
            
        log_callback("Storing in vector database...")
        vector_store.add_chunks(
            document_id=document_id,
            document_name=document_name,
            chunks=all_chunks,
            embeddings=all_embeddings,
            pages=all_pages
        )
        log_callback("Indexing complete.")
    else:
        log_callback(f"Unsupported file format for RAG indexing: {file_path.suffix}")
