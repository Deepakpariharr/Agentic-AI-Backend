"""Limits on what a chat request may carry, shared by every chat endpoint
(POST /chat/mcpAgentChat and POST /chat/documents).

Each check returns an error message for the caller to send back, or None when
the input is fine, so every endpoint can keep its own response format.
"""
from typing import Optional

from fastapi import UploadFile

MAX_QUERY_CHARS = 4000
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_FILES = 3
ALLOWED_EXTENSION = ".pdf"


# This is Like Frontend should call before sending the input to the backend in that case we can provide the input to them
def limits() -> dict:
    """The limits above, as served by GET /chat/limits so the frontend can
    check input before sending it."""
    return {
        "maxQueryChars": MAX_QUERY_CHARS,
        "maxFileSizeBytes": MAX_FILE_SIZE_BYTES,
        "maxFiles": MAX_FILES,
        "allowedFileTypes": [ALLOWED_EXTENSION],
    }

def validate_query(text:str) -> Optional[str]:
    if not text.strip():
        return "Message must not be empty"
    if len(text) > MAX_QUERY_CHARS:
        return f"Message must be at most {MAX_QUERY_CHARS} characters"
    return None





async def validate_upload(file: UploadFile) -> Optional[str]:
    if not file.filename or not file.filename.lower().endswith(ALLOWED_EXTENSION):
        return "Only PDF files are allowed"

    size = file.size
    if size is None:  # not set by the multipart parser, so measure it
        size = len(await file.read())
        await file.seek(0)
    if size > MAX_FILE_SIZE_BYTES:
        return f"'{file.filename}' is larger than 1 MB"
    return None


async def validate_uploads(files: list[UploadFile]) -> Optional[str]:
    """Check a multi-file upload: at most MAX_FILES, each passing validate_upload."""
    if len(files) > MAX_FILES:
        return f"Upload at most {MAX_FILES} files at a time"
    for file in files:
        if error := await validate_upload(file):
            return error
    return None
