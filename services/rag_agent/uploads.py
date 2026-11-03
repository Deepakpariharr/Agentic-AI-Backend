"""Uploaded PDFs → text → the user's RAG index."""
import hashlib
import logging
import re
from typing import List

import fitz  # PyMuPDF
from fastapi import UploadFile

from services import document_registry
from services.rag import RAGService

logger = logging.getLogger(__name__)

MAX_CHARACTER_ALLOWED=75000



# Ingesting files into the Rag service
async def ingest_pdfs(rag: RAGService, files: list[UploadFile], session_id: str) -> bool:
    """Index every readable PDF in `files` under `session_id`.

    Non-PDF, empty and text-less files are skipped. Returns True if at least
    one file made it into the index.
    """
    any_ingested = False
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            logger.warning(f"Skipping non-PDF file: {file.filename}")
            continue

        await file.seek(0)
        content = await file.read()
        if not content:
            logger.warning(f"Skipping empty file: {file.filename}")
            continue
        
        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                full_text = "\n\n".join(page.get_text() for page in pdf).strip()

            if not full_text:
                logger.warning(f"No extractable text found in {file.filename}")
                continue
            
            full_text=full_text.strip()
            
            if len(full_text) > MAX_CHARACTER_ALLOWED:
                logger.warning(f"Skipping {file.filename}: more than {MAX_CHARACTER_ALLOWED} characters of text")
                continue

            file_hash = hashlib.sha256(content).hexdigest()  # same file bytes → same id

            # Postgres bookkeeping (no-op without a database): skip a file this
            # user already has instead of embedding and storing it twice.
            if not await document_registry.register(session_id, file_hash, file.filename, len(content)):
                logger.info(f"Skipping {file.filename}: already uploaded by session {session_id}")
                continue

            try:
                result = await rag.ingest_document(
                    raw_text=full_text,
                    session_id=session_id,
                    source_id=file_hash,
                    file_name=file.filename,
                )
            except Exception as e:
                await document_registry.mark_failed(session_id, file_hash, str(e))
                raise

            if result and "error" not in str(result).lower() and "failed" not in str(result).lower():
                any_ingested = True
                chunks = re.search(r"(\d+) chunks", str(result))
                await document_registry.mark_ready(session_id, file_hash, int(chunks.group(1)) if chunks else None)
                logger.info(f"Ingested {file.filename} for session {session_id}: {result}")
            else:
                await document_registry.mark_failed(session_id, file_hash, str(result))
                logger.error(f"Failed to ingest {file.filename}: {result}")

        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}", exc_info=True)

    return any_ingested
