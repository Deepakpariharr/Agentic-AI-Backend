from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    session_id: str = Field(..., description="Strict tenant isolation ID")
    source_id: str = Field(..., description="Unique identifier for the uploaded file")
    chunk_index: int = Field(..., description="Sequential index to preserve document order")
    document_type: str = Field(default="general", description="e.g., pdf, csv, text")
