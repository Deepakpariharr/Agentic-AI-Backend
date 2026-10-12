"""Uploaded PDFs → text → the user's RAG index."""
import logging
from typing import List

import fitz  # PyMuPDF
from fastapi import UploadFile

from services.rag import RAGService

logger = logging.getLogger(__name__)


async def ingest_pdfs(rag: RAGService, files: List[UploadFile], session_id: str) -> bool:
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

            result = await rag.ingest_document(
                raw_text=full_text,
                session_id=session_id,
                source_id=file.filename,
            )

            if result and "error" not in str(result).lower() and "failed" not in str(result).lower():
                any_ingested = True
                logger.info(f"Ingested {file.filename} for session {session_id}: {result}")
            else:
                logger.error(f"Failed to ingest {file.filename}: {result}")

        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}", exc_info=True)

    return any_ingested
