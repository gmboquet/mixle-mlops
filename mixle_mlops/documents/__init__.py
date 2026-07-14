"""Document ingestion: extract text from uploaded files and chunk it for embedding/retrieval."""
from __future__ import annotations

from .parse import (
    DocumentParseError,
    LocatedChunk,
    chunk_text,
    extract_text,
    extract_typed_tables,
    parse_and_chunk,
    parse_and_chunk_located,
)

__all__ = [
    "DocumentParseError",
    "LocatedChunk",
    "extract_text",
    "chunk_text",
    "parse_and_chunk",
    "parse_and_chunk_located",
    "extract_typed_tables",
]
