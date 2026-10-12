"""utility/request_limits.py: the query and upload limits both chat endpoints share."""
import io

import pytest
from fastapi import UploadFile

from utility.request_limits import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILES,
    MAX_QUERY_CHARS,
    validate_query,
    validate_upload,
    validate_uploads,
)


def _upload(name="resume.pdf", size=10, known_size=True):
    """An UploadFile of `size` bytes. `known_size=False` mimics a file whose
    size the multipart parser didn't record."""
    return UploadFile(
        file=io.BytesIO(b"0" * size),
        filename=name,
        size=size if known_size else None,
    )


# ── validate_query ───────────────────────────────────────────────────────────

def test_query_within_limit_is_accepted():
    assert validate_query("x" * MAX_QUERY_CHARS) is None


def test_query_over_limit_is_rejected():
    assert validate_query("x" * (MAX_QUERY_CHARS + 1)) == "Message must be at most 4000 characters"


def test_blank_query_is_rejected_by_default():
    assert validate_query("   ") == "Message should not be blank"


def test_blank_query_is_accepted_when_allowed():
    assert validate_query("", allow_blank=True) is None


# ── validate_upload ──────────────────────────────────────────────────────────

async def test_pdf_within_limit_is_accepted():
    assert await validate_upload(_upload(size=MAX_FILE_SIZE_BYTES)) is None


@pytest.mark.parametrize("name", ["resume.docx", "resume.pdf.exe", "", None])
async def test_non_pdf_is_rejected(name):
    assert await validate_upload(_upload(name=name)) == "Only PDF files are allowed"


async def test_pdf_extension_check_ignores_case():
    assert await validate_upload(_upload(name="RESUME.PDF")) is None


async def test_pdf_over_limit_is_rejected():
    error = await validate_upload(_upload(size=MAX_FILE_SIZE_BYTES + 1))
    assert error == "'resume.pdf' is larger than 1 MB"


async def test_unknown_size_is_measured_and_file_rewound():
    upload = _upload(size=MAX_FILE_SIZE_BYTES + 1, known_size=False)

    assert await validate_upload(upload) == "'resume.pdf' is larger than 1 MB"
    assert len(await upload.read()) == MAX_FILE_SIZE_BYTES + 1  # still readable


# ── validate_uploads ─────────────────────────────────────────────────────────

async def test_up_to_max_files_are_accepted():
    assert await validate_uploads([_upload() for _ in range(MAX_FILES)]) is None


async def test_too_many_files_are_rejected():
    error = await validate_uploads([_upload() for _ in range(MAX_FILES + 1)])
    assert error == f"Upload at most {MAX_FILES} files at a time"


async def test_one_bad_file_rejects_the_upload():
    files = [_upload(), _upload(name="notes.txt")]
    assert await validate_uploads(files) == "Only PDF files are allowed"
