import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_ENCODING_NAME, CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

from .models import DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentIngestor:
    def __init__(self):
        # chunk_size / chunk_overlap are measured in tokens, not characters.
        # other method also availalble for chunking the data like overlapping, fixed size, regression based ,semantic chunking, llm chunking 
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=CHUNK_ENCODING_NAME,
            chunk_size=CHUNK_SIZE_TOKENS,
            chunk_overlap=CHUNK_OVERLAP_TOKENS,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        )

    def process_document(
        self,
        text: str,
        session_id: str,
        source_id: str,
        doc_type: str = "general",
    ) -> list[Document]:
        if not text or not text.strip():
            return []

        raw_chunks = self.text_splitter.split_text(text)
        validated_docs = []

        for index, chunk_text in enumerate(raw_chunks):
            meta = DocumentMetadata(
                session_id=session_id,   # used for like which document is owned by which user two users dont sees each others data
                source_id=source_id,    # file name is called here the source Id 
                chunk_index=index,    # cuhnk position together with doc it forms citations like resume.pdf#3, resume.pdf#4 kind of 
                document_type=doc_type, #  what kind of documents i si t in general or something else tag
            )
            validated_docs.append(
                Document(page_content=chunk_text, metadata=meta.model_dump())
            )

        logger.info(f"Chunked {source_id} into {len(validated_docs)} segments.")
        return validated_docs