"""calculate: arithmetic the model shouldn't do in its head."""
import math
from typing import Annotated

from pydantic import Field


def calculate(
    expression: Annotated[
        str,
        Field(description="Arithmetic using numbers, + - * / and parentheses, e.g. '(120 + 80) / 4'."),
    ],
) -> str:
    """Evaluate an arithmetic expression and return the result.

    Supports + - * / and parentheses only. Use this for any calculation
    (totals, percentages, conversions) instead of working it out yourself.
    """
    try:
        # Safety first: allow only numbers, math ops, and decimal points.
        # `**` is refused too: 9**9**9**9 would hang the tool server.
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression) or "**" in expression:
            return "❌ Invalid expression. Only numbers and basic math operators allowed."

        result = eval(expression, {"__builtins__": None}, {"math": math})
        return str(result)
    except Exception as e:
        return f"⚠️ Error evaluating expression: {e}"
