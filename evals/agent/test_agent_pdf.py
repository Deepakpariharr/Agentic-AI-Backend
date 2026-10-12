"""PDF upload evals.

agent_response extracts the PDF text itself and appends it to the prompt, so
the agent should answer from that text without inventing anything that isn't
in the document.
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

import eval_metrics as m
from harness import make_pdf, new_user_id, run_agent

INVOICE_TEXT = """Invoice #INV-2041
Customer: Asha Verma
Invoice date: 12 March 2026
Items:
  2 x Wireless Mouse @ 850 INR each
  1 x USB-C Hub @ 2300 INR
Total due: 4000 INR
Payment due by: 11 April 2026"""

RESUME_TEXT = """Priya Nair - Backend Developer
Experience: 2 years at Zeta Labs building REST APIs with FastAPI and PostgreSQL.
Skills: Python, FastAPI, Docker, AWS Lambda, SQL
Education: B.Tech Computer Science, 2023
Location: Pune, India"""

PDF_CASES = [
    pytest.param(
        INVOICE_TEXT,
        "Who is the customer on this invoice and what is the total due?",
        "The customer is Asha Verma and the total due is 4000 INR.",
        id="invoice-lookup",
    ),
    pytest.param(
        INVOICE_TEXT,
        "When is payment due?",
        "Payment is due by 11 April 2026.",
        id="invoice-date",
    ),
    pytest.param(
        INVOICE_TEXT,
        "Is the total on this invoice correct?",
        "Yes. 2 × 850 + 2300 = 4000 INR, which matches the stated total.",
        id="invoice-arithmetic",
    ),
    pytest.param(
        RESUME_TEXT,
        "Summarise this resume in one sentence.",
        "Priya Nair is a Pune-based backend developer with 2 years of experience "
        "building FastAPI/PostgreSQL REST APIs at Zeta Labs.",
        id="resume-summary",
    ),
    pytest.param(
        RESUME_TEXT,
        "What is this candidate's phone number?",
        "The resume does not include a phone number.",
        id="resume-missing-field",
    ),
]


@pytest.mark.pdf
@pytest.mark.parametrize("pdf_text, prompt, expected", PDF_CASES)
def test_pdf_answers(pdf_text, prompt, expected):
    run = run_agent(prompt, user_id=new_user_id(), pdf=make_pdf(pdf_text))

    test_case = LLMTestCase(
        input=prompt,
        actual_output=run.output,
        expected_output=expected,
        retrieval_context=[pdf_text],
        tools_called=run.tools_called,
    )
    assert_test(test_case, [m.correctness(), m.faithfulness()])
