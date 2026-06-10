from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from app.core.config import settings
from app.services.llm import llm_client
from app.services.model_tasks import embed_texts
from app.services.vector_store import vector_store
from app.db.session import get_session

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
    
    all_chunks = []
    all_pages = []

    if file_path.suffix.lower() == ".pdf":
        log_callback("Opening PDF file...")
        try:
            reader = PdfReader(file_path)
        except Exception as e:
            log_callback(f"Failed to open PDF: {e}")
            return
            
        log_callback("Extracting and chunking text from PDF...")
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                page_chunks = chunk_text(text)
                all_chunks.extend(page_chunks)
                all_pages.extend([str(page_num + 1)] * len(page_chunks))
                
    elif file_path.suffix.lower() == ".docx":
        log_callback("Opening DOCX file...")
        try:
            from docx import Document
            doc = Document(file_path)
        except Exception as e:
            log_callback(f"Failed to open DOCX: {e}")
            return
            
        log_callback("Extracting and chunking text from DOCX...")
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
                
        text = "\n".join(full_text)
        if text.strip():
            page_chunks = chunk_text(text)
            all_chunks.extend(page_chunks)
            all_pages.extend(["1"] * len(page_chunks))
            
    elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        log_callback("Opening Image file for Vision OCR...")
        all_chunks = [] # Will be populated by Vision section below
            
    else:
        log_callback(f"Unsupported file format for RAG indexing: {file_path.suffix}")
        return

    if all_chunks and file_path.suffix.lower() == ".pdf":
        total_extracted_len = sum(len(c) for c in all_chunks)
        full_text = " ".join(all_chunks)
        
        import re
        letters_and_numbers = len(re.findall(r'[a-zA-Zа-яА-ЯёЁ0-9]', full_text))
        
        if total_extracted_len < len(reader.pages) * 50 or letters_and_numbers < total_extracted_len * 0.5:
            log_callback(f"Extracted [PDF] text appears incomplete or garbled. Trying OCR fallback...")
            all_chunks = []
            all_pages = []

    if not all_chunks:
        log_callback("No valid text found in document. Trying Vision-LLM OCR...")
        try:
            import base64
            import io
            from app.services.llm import llm_client
            
            images_to_process = []
            
            if file_path.suffix.lower() == ".pdf":
                from pdf2image import convert_from_path
                log_callback("Converting PDF to images for Vision analysis...")
                images_to_process = convert_from_path(file_path, dpi=120)
            elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                from PIL import Image
                images_to_process = [Image.open(file_path)]
            else:
                log_callback(f"OCR fallback is not supported for {file_path.suffix} files yet.")
                return

            import time
            for i, page in enumerate(images_to_process):
                if i >= 20:
                    log_callback("Economy Mode: Stopping OCR at page 20.")
                    break
                    
                log_callback(f"OCR: Analyzing page {i + 1}/{len(images_to_process)} using {settings.groq_vision_model}...")
                
                # Convert PIL Image to base64
                img_byte_arr = io.BytesIO()
                page.convert('RGB').save(img_byte_arr, format='JPEG', quality=75)
                base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{base64_image}"
                
                try:
                    res = await llm_client.ocr_image(data_url)
                    extracted_text = res.content
                    if extracted_text and extracted_text.strip():
                        page_chunks = chunk_text(extracted_text)
                        all_chunks.extend(page_chunks)
                        all_pages.extend([str(i + 1)] * len(page_chunks))
                    
                    # Small delay to respect Groq rate limits (free tier)
                    time.sleep(1.0)
                except Exception as page_exc:
                    log_callback(f"Vision failed on page {i+1}: {page_exc}")
                    if "429" in str(page_exc):
                        log_callback("Rate limit reached. Waiting 15s...")
                        time.sleep(15)
                    continue

        except Exception as e:
            log_callback(f"Vision OCR failed: {e}")

        if not all_chunks:
            log_callback("No text found even after Vision OCR attempt.")
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
            dimension = settings.embedding_dimensions
            all_embeddings.extend([[0.0] * dimension for _ in batch_texts])
        
    log_callback("Storing in vector database...")
    vector_store.add_chunks(
        document_id=document_id,
        document_name=document_name,
        chunks=all_chunks,
        embeddings=all_embeddings,
        pages=all_pages
    )

    # Economical NER: Analyze only first 2 chunks
    from app.services.model_tasks import extract_geo_knowledge
    log_callback("Extracting geological entities for GraphRAG...")
    
    entities_count = 0
    for i, chunk in enumerate(all_chunks[:2]):
        try:
            res = await extract_geo_knowledge(chunk, source_id=document_id)
            if res.entities or res.relations:
                with get_session() as session:
                    from app.db.repository import graph_repo
                    graph_repo.add_extracted_knowledge(
                        session, 
                        document_id, 
                        res.entities, 
                        res.relations
                    )
                entities_count += len(res.entities)
        except Exception as e:
            log_callback(f"Graph extraction failed for chunk {i}: {e}")
            
    log_callback(f"Indexing complete. Extracted {entities_count} entities for Knowledge Graph.")
