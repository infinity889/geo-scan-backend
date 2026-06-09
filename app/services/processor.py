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
            
    else:
        log_callback(f"Unsupported file format for RAG indexing: {file_path.suffix}")
        return

    if all_chunks and file_path.suffix.lower() == ".pdf":
        total_extracted_len = sum(len(c) for c in all_chunks)
        full_text = " ".join(all_chunks)
        
        import re
        letters_and_numbers = len(re.findall(r'[a-zA-Zа-яА-ЯёЁ0-9]', full_text))
        
        if total_extracted_len < len(reader.pages) * 50 or letters_and_numbers < total_extracted_len * 0.5:
            log_callback(f"Extracted text appears incomplete or garbled (Len: {total_extracted_len}, Valid: {letters_and_numbers}). Trying OCR fallback...")
            all_chunks = []
            all_pages = []

    if not all_chunks:
        log_callback("No valid text found in document. Trying OCR fallback with PaddleOCR...")
        try:
            import os
            from pdf2image import convert_from_path
            from paddleocr import PaddleOCR
            import numpy as np
            
            project_dir = Path(__file__).resolve().parent.parent.parent
            bin_dir = project_dir / "bin"
            poppler_path = str(bin_dir / "poppler" / "Library" / "bin")
            
            if os.path.exists(poppler_path):
                log_callback("Converting PDF to images...")
                pages = convert_from_path(file_path, dpi=200, poppler_path=poppler_path)
                
                log_callback(f"Found {len(pages)} pages. Initializing PaddleOCR...")
                ocr_model = PaddleOCR(use_angle_cls=True, lang='ru', show_log=False)
                
                for i, page in enumerate(pages):
                    log_callback(f"Running OCR on page {i + 1}/{len(pages)}...")
                    
                    # Convert PIL Image to numpy array
                    img_array = np.array(page)
                    
                    # Run PaddleOCR
                    result = ocr_model.ocr(img_array, cls=True)
                    
                    page_text = []
                    if result:
                        for res in result:
                            if res:
                                for line in res:
                                    if len(line) == 2 and len(line[1]) >= 1:
                                        text = line[1][0]
                                        page_text.append(text)
                    
                    full_text = "\n".join(page_text)
                    if full_text and full_text.strip():
                        page_chunks = chunk_text(full_text)
                        all_chunks.extend(page_chunks)
                        all_pages.extend([str(i + 1)] * len(page_chunks))
            else:
                log_callback("Poppler binary not found. Please check bin directory.")
        except Exception as e:
            log_callback(f"OCR failed: {e}")

        if not all_chunks:
            log_callback("No text found even after OCR attempt.")
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
            dimension = 32
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
