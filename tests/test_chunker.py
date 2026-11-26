"""Chunker: parameter guards, UTF-8 safety at token boundaries, coverage."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import tiktoken

from knowledge_palace.ingest.chunker import chunk_fixed, chunk_semantic, chunk_text

ENC = tiktoken.get_encoding("cl100k_base")

CJK = ("非同期処理ではタスクをポーリングするまで何も実行されません。"
       "実行系がワーカースレッドにタスクを割り当てます。" * 200)
EMOJI = "🎉🚀✨ party time with flags 🇯🇵🇺🇸 and families 👨‍👩‍👧‍👦 everywhere " * 300
ASCII = ENC.decode(ENC.encode("The quick brown fox jumps over the lazy dog. ") * 600)


def _chunk_in_subprocess(text: str, max_tokens: int, overlap_tokens: int, timeout: float = 10.0) -> tuple[str, str]:
    """Run chunk_fixed in a throwaway interpreter; return ("ok"|"error"|"hung", detail).

    subprocess.run() kills the child on timeout, so a chunker stuck in an
    infinite loop surfaces as ("hung", ...) instead of deadlocking the suite
    (a ProcessPoolExecutor would join the looping worker forever).
    """
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from knowledge_palace.ingest.chunker import chunk_fixed\n"
        "cs = chunk_fixed(sys.argv[2], max_tokens=int(sys.argv[3]), overlap_tokens=int(sys.argv[4]))\n"
        "print('N', len(cs))\n"
    )
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", code, str(_SRC_ROOT), text, str(max_tokens), str(overlap_tokens)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return "hung", ""
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        return "error", lines[-1] if lines else ""
    return "ok", proc.stdout.strip()


# The isolated child ignores PYTHONPATH, so the tree under test is passed explicitly;
# KP_TEST_SRC_ROOT lets a run point the child at a different checkout (fail-before runs).
_SRC_ROOT = os.environ.get("KP_TEST_SRC_ROOT") or \
    str(Path(__file__).resolve().parent.parent / "src")


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens"),
    [(512, 512), (512, 600), (50, 50), (0, 50), (-5, 50), (512, -10)],
)
def test_degenerate_windows_are_rejected_instead_of_hanging(max_tokens: int, overlap_tokens: int):
    # At the audited revision the first five of these never returned (the loop
    # step was <= 0) and the last silently skipped text between chunks.
    outcome, detail = _chunk_in_subprocess(ASCII, max_tokens, overlap_tokens)
    assert outcome == "error", f"chunk_fixed({max_tokens}, {overlap_tokens}) {outcome}: {detail}"
    assert "overlap_tokens|max_tokens" in detail or "overlap_tokens must" in detail or "max_tokens must" in detail


def test_valid_window_still_works_in_a_subprocess():
    outcome, detail = _chunk_in_subprocess(ASCII, 512, 50)
    assert outcome == "ok", detail
    assert int(detail.split()[1]) > 5


def test_guard_also_covers_the_semantic_strategy():
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_semantic(ASCII, max_tokens=512, overlap_tokens=512)
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_text(ASCII, strategy="semantic", max_tokens=512, overlap_tokens=512)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_text("hello", strategy="sliding")


@pytest.mark.parametrize("text", [CJK, EMOJI, CJK + "\n\n" + EMOJI], ids=["cjk", "emoji", "mixed"])
def test_multibyte_text_is_never_cut_mid_character(text: str):
    # BPE tokens can hold part of a CJK character or an emoji; decoding such a
    # slice yields U+FFFD. Before the fix 4 of 23 CJK chunks and 14 of 29 emoji
    # chunks contained replacement characters that were not in the source.
    assert "\ufffd" not in text
    for overlap in (0, 1, 50):
        chunks = chunk_fixed(text, max_tokens=512, overlap_tokens=overlap)
        broken = [c.index for c in chunks if "\ufffd" in c.content]
        assert broken == [], f"overlap={overlap}: U+FFFD in chunks {broken}"
        assert all(c.content for c in chunks)


def test_every_chunk_is_verbatim_and_chunks_cover_the_whole_text():
    text = CJK + "\n\n" + EMOJI
    chunks = chunk_fixed(text, max_tokens=512, overlap_tokens=50)
    starts = []
    for chunk in chunks:
        assert chunk.content in text, f"chunk {chunk.index} is not a substring of the source"
        starts.append(text.index(chunk.content))
    assert starts[0] == 0, "the first chunk must start at the beginning"
    # Each chunk starts before the previous one ends: no gap anywhere.
    for previous, current in zip(chunks, starts, strict=True):
        assert current <= text.index(previous.content) + len(previous.content)
    assert text.endswith(chunks[-1].content), "the last chunk must reach the end of the text"


def test_chunks_respect_the_token_budget_and_end_of_text():
    total = len(ENC.encode(ASCII))
    chunks = chunk_fixed(ASCII, max_tokens=512, overlap_tokens=50)
    assert all(c.token_count <= 512 for c in chunks)
    assert chunks[-1].metadata["end_token"] == total
    # Re-assembling without the overlap reproduces the source exactly.
    rebuilt = chunks[0].content
    for previous, chunk in zip(chunks, chunks[1:], strict=False):
        rebuilt += chunk.content[len(ENC.decode(ENC.encode(previous.content)[462:])):]
    assert rebuilt == ASCII


def test_books_without_headings_fall_back_to_fixed_windows():
    # Extracted books carry no markdown headings, so the "semantic" strategy
    # degenerates to fixed windows for them.
    chunks = chunk_semantic(("An executor polls futures. " * 1500), max_tokens=512, overlap_tokens=50)
    assert {c.metadata["strategy"] for c in chunks} == {"semantic-sub"}
    assert all(c.token_count <= 512 for c in chunks)


def test_notes_with_headings_are_split_on_them():
    notes = "# A\n\n" + ("alpha beta " * 900) + "\n\n# B\n\n" + ("gamma delta " * 40) + "\n"
    chunks = chunk_semantic(notes, max_tokens=512, overlap_tokens=50, respect_headings=True)
    assert chunks[0].content.startswith("# A")
    last = chunks[-1]
    assert last.content.startswith("# B") and last.metadata["section_start"] is True
    assert last.metadata["strategy"] == "semantic"


@pytest.mark.parametrize("text", ["", "   ", "\n\t "], ids=["empty", "spaces", "whitespace"])
def test_blank_text_yields_no_chunks(text: str):
    for strategy in ("semantic", "fixed"):
        assert chunk_text(text, strategy=strategy) == []
