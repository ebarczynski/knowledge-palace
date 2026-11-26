"""Text chunking strategies for embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

# Module-level cache: avoid repeated tiktoken.get_encoding() calls which are
# expensive when processing hundreds of books with thousands of chunks.
_encoding_name: str = "cl100k_base"
_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    """Return the cached tiktoken encoding, initialising on first call."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(_encoding_name)
    return _encoding


@dataclass
class Chunk:
    """A chunk of text ready for embedding."""
    content: str
    index: int
    token_count: int
    metadata: dict


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    enc = _get_encoding() if model == _encoding_name else tiktoken.get_encoding(model)
    return len(enc.encode(text, disallowed_special=()))


def _validate_window(max_tokens: int, overlap_tokens: int) -> None:
    """Reject window parameters that would stall the loop or skip text.

    ``chunk_fixed`` advances by ``max_tokens - overlap_tokens`` per chunk, so a
    step below one token never terminates and a negative one drops text.
    """
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"max_tokens must be a positive integer, got {max_tokens!r}")
    if not isinstance(overlap_tokens, int) or not 0 <= overlap_tokens < max_tokens:
        raise ValueError(
            f"overlap_tokens must satisfy 0 <= overlap_tokens < max_tokens, got {overlap_tokens!r}"
        )


def _utf8_aligned_bounds(tok_bytes: list[bytes], start: int, end: int) -> tuple[int, int]:
    """Widen a ``[start, end)`` token range to one that decodes as valid UTF-8.

    BPE tokens can hold part of a multi-byte character (common for CJK and
    emoji). Decoding a range that cuts such a character yields U+FFFD, so the
    corrupt character would be stored and embedded. The start moves back to the
    first token that begins a character and the end moves in past an unfinished
    one; with any overlap the neighbouring chunk carries the rest.
    """
    lo = start
    while lo > 0 and _is_continuation_byte(tok_bytes[lo][0]):
        lo -= 1
    hi = end
    while hi > lo:
        try:
            b"".join(tok_bytes[lo:hi]).decode("utf-8")
            return lo, hi
        except UnicodeDecodeError:
            hi -= 1
    # A lone token that is not a complete character (only possible with a
    # window of one token): keep the nominal cut rather than emit nothing.
    return start, end


def _is_continuation_byte(byte: int) -> bool:
    return 0x80 <= byte <= 0xBF


def chunk_fixed(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Split text into fixed-size overlapping chunks."""
    _validate_window(max_tokens, overlap_tokens)
    enc = _get_encoding()
    tokens = enc.encode(text, disallowed_special=())
    tok_bytes = [enc.decode_single_token_bytes(t) for t in tokens]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max_tokens - overlap_tokens

    while start < len(tokens):
        lo, hi = _utf8_aligned_bounds(tok_bytes, start, min(start + max_tokens, len(tokens)))
        chunks.append(Chunk(
            content=b"".join(tok_bytes[lo:hi]).decode("utf-8"),
            index=idx,
            token_count=hi - lo,
            metadata={"strategy": "fixed", "start_token": lo, "end_token": hi},
        ))

        start += step
        idx += 1

        # A tail shorter than the overlap is already covered by this chunk.
        if len(tokens) - start < overlap_tokens:
            break

    return chunks


def chunk_semantic(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    respect_headings: bool = True,
) -> list[Chunk]:
    """Split text by semantic boundaries (headings, paragraphs), then by size."""
    _validate_window(max_tokens, overlap_tokens)
    sections: list[str] = []

    if respect_headings:
        # Split on markdown headings
        heading_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
        parts = heading_pattern.split(text)

        current_section = ""
        for part in parts:
            if heading_pattern.match(part):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = part + "\n"
            else:
                current_section += part
        if current_section.strip():
            sections.append(current_section.strip())
    else:
        # Split on double newlines (paragraphs)
        sections = [s.strip() for s in text.split("\n\n") if s.strip()]

    if not sections:
        sections = [text]

    # Now split oversized sections into fixed chunks
    chunks: list[Chunk] = []
    idx = 0

    for section in sections:
        token_count = count_tokens(section)
        if token_count <= max_tokens:
            chunks.append(Chunk(
                content=section,
                index=idx,
                token_count=token_count,
                metadata={"strategy": "semantic", "section_start": True},
            ))
            idx += 1
        else:
            # Section too large, sub-chunk it
            sub_chunks = chunk_fixed(section, max_tokens, overlap_tokens)
            for sc in sub_chunks:
                chunks.append(Chunk(
                    content=sc.content,
                    index=idx,
                    token_count=sc.token_count,
                    metadata={"strategy": "semantic-sub", "section_start": sc.index == 0},
                ))
                idx += 1

    return chunks


def chunk_text(
    text: str,
    strategy: str = "semantic",
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    respect_headings: bool = True,
) -> list[Chunk]:
    """Chunk text using the specified strategy."""
    if not text.strip():
        return []

    if strategy == "semantic":
        return chunk_semantic(text, max_tokens, overlap_tokens, respect_headings)
    if strategy == "fixed":
        return chunk_fixed(text, max_tokens, overlap_tokens)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
