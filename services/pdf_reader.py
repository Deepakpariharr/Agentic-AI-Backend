import base64

import fitz  # PyMuPDF


def read_pdf(file_content: str) -> str:
    try:

        file_bytes = base64.b64decode(file_content)

        pdf = fitz.open(stream=file_bytes, filetype="pdf")

        text = ""
        for page in pdf:
            text += page.get_text("text")

        return text or "No readable text found in the PDF."

    except Exception as e:  # noqa: BLE001
        return f"Error reading PDF: {str(e)}"
