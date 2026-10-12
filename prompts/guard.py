"""Prompt-injection guard shared by the RAG pipeline and the MCP tools."""
import re

# Matches an opening or closing <untrusted_data ...> tag, any casing/spacing.
_GUARD_TAG_RE = re.compile(r"<(\s*/?\s*untrusted_data)", re.IGNORECASE)


def wrap_untrusted(source: str, body: str) -> str:
    """Wrap third-party text so the model treats it as data, not instructions.

    Any guard tag already inside `body` is neutralised so content cannot close
    the block early and smuggle text out as if it were trusted.
    """
    safe_body = _GUARD_TAG_RE.sub(r"&lt;\1", body)
    return (
        f'<untrusted_data source="{source}">\n'
        f"{safe_body}\n"
        f"</untrusted_data>"
    )
