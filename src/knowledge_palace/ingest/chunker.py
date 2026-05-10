"""Text chunking strategies for embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken


@dataclass
class Chunk:
    """A chunk of text ready for embedding."""
    content: str
    index: int
    token_count: int
    metadata: dict


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def chunk_fixed(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Split text into fixed-size overlapping chunks."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        content = enc.decode(chunk_tokens)

        chunks.append(Chunk(
            content=content,
            index=idx,
            token_count=len(chunk_tokens),
            metadata={"strategy": "fixed", "start_token": start, "end_token": end},
        ))

        start += max_tokens - overlap_tokens
        idx += 1

        # Avoid tiny trailing chunks
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
    elif strategy == "fixed":
        return chunk_fixed(text, max_tokens, overlap_tokens)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
