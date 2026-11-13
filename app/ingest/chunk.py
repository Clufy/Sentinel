"""Chunking: one chunk per control; split only if summary exceeds the token cap.

Tokens are approximated as len(text.split()) per the locked design.
"""


def chunk_control(title: str, summary: str, max_chunk_tokens: int = 512) -> list[str]:
    words = summary.split()
    if len(words) <= max_chunk_tokens:
        # Preserve original whitespace when it fits in one chunk.
        return [summary]
    return [
        " ".join(words[i : i + max_chunk_tokens])
        for i in range(0, len(words), max_chunk_tokens)
    ]


def embed_input(title: str, chunk_text: str) -> str:
    """Text that is embedded for a chunk: the title prefixed to the chunk text."""
    return f"{title}. {chunk_text}"
