# packages/retriever/chunking.py
"""
Text chunking utilities — token-aware, table-aware, tiny, and explicit.

What this module provides
-------------------------
- count_tokens(text: str, model: str | None = None) -> int
    Best-effort token count using `tiktoken` if installed; else a char→token heuristic.

- split_by_headings(text: str) -> list[str]
    Lightweight splitter that detects common finance headings (ALL CAPS, 1.1., A., etc.).

- semantic_paragraphs(text: str) -> list[str]
    Conservative paragraph splitter (double-newlines + bullet blocks kept together).

- chunk_text(
      text: str,
      *,
      target_tokens: int = 512,
      overlap_tokens: int = 64,
      model: str | None = None,
      title_prefix: str | None = None,
      hard_max_tokens: int | None = 800,
  ) -> list[dict]
    Token-aware sliding window chunker with optional title-prefixing for better retrieval.
    Returns a list of dicts:
      {"text": "...", "position": <int>, "tokens": <int>, "title": <str|None>}

- chunk_table(df: pandas.DataFrame, *, max_rows_per_chunk: int = 30) -> list[dict]
    Turn a table into row-grouped chunks with TSV serialization and a small header.

Design goals
------------
- Keep behavior **predictable** and tutorial-clear.
- Avoid hidden state; every function is pure and easy to test.
- Favor stability over aggressive splitting (finance text often has dense lists/tables).

Typical usage
-------------
from packages.retriever.chunking import chunk_text

chunks = chunk_text(raw_text, target_tokens=400, overlap_tokens=50, title_prefix="Q4 2024 Report")
for c in chunks:
    print(c["position"], c["tokens"], c["text"][:80])

Notes
-----
- If you need super-precise tokenization, install `tiktoken` and pass a model name
  (e.g., "gpt-4o-mini" or "text-embedding-3-small"). Otherwise, a ~4 chars per token
  heuristic is used.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Optional dependency (kept lazy)
try:  # pragma: no cover
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore


# ---------------------------
# Token counting
# ---------------------------

def _heuristic_token_count(text: str) -> int:
    """
    Rough heuristic: ~4 characters per token (OpenAI-ish).
    """
    n = len(text)
    return max(1, n // 4)


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Return the number of tokens in `text`. Uses tiktoken if available.

    Args:
      text  : input string
      model : tiktoken encoding model (optional). If None, chooses a default.

    Returns:
      int token count (>=1)
    """
    if not text:
        return 1

    if tiktoken is None:
        return _heuristic_token_count(text)

    try:
        if not model:
            # Fallback to a common encoding; "cl100k_base" works for GPT-3.5/4 families
            enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.encoding_for_model(model)  # type: ignore[attr-defined]
        return len(enc.encode(text))
    except Exception:
        return _heuristic_token_count(text)


# ---------------------------
# Primitive splitters
# ---------------------------

_HEADING_RE = re.compile(
    r"""
    ^\s*(
        (?:
            [A-Z][A-Z0-9 &/\-]{2,}        # ALL CAPS headings (e.g., "MANAGEMENT DISCUSSION")
            |
            (?:\d{1,2}\.){1,4}\s+.*       # numbered 1. / 1.1.1. sections
            |
            [A-Z]\.\s+.*                  # lettered A. B. C.
            |
            Appendix\s+[A-Z0-9]+.*        # Appendix markers
        )
    )\s*$
    """,
    re.VERBOSE,
)


def split_by_headings(text: str) -> List[str]:
    """
    Split text into sections when a heading-like line is encountered.

    - Heuristics tuned for filings and reports (ALL CAPS, numbered, lettered).
    - Preserves headings as part of the following section.
    - If no headings are found, returns a single-element list [text].

    This is intentionally conservative.
    """
    if not text.strip():
        return [text]

    lines = text.splitlines()
    blocks: List[str] = []
    buf: List[str] = []

    def _flush():
        if buf:
            blocks.append("\n".join(buf).strip())
            buf.clear()

    for ln in lines:
        if _HEADING_RE.match(ln.strip()):
            # start of a new section
            _flush()
            buf.append(ln)
        else:
            buf.append(ln)
    _flush()

    # Remove empties
    return [b for b in blocks if b and b.strip()]


def semantic_paragraphs(text: str) -> List[str]:
    """
    Conservative paragraph splitter:
      - Split on double newlines
      - Keep bullet blocks intact (lines starting with '-', '*', '•')
      - Trim whitespace around paragraphs
    """
    if not text.strip():
        return [text]

    raw_parts = [p.strip() for p in re.split(r"\n{2,}", text) if p and p.strip()]
    parts: List[str] = []
    buf: List[str] = []

    def starts_bullet(s: str) -> bool:
        return bool(re.match(r"^\s*[\-\*\u2022]\s+", s))

    for p in raw_parts:
        lines = p.splitlines()
        # If a paragraph is primarily bullets, keep as one
        if sum(1 for ln in lines if starts_bullet(ln)) >= max(1, len(lines) // 2):
            if buf:
                parts.append("\n".join(buf).strip())
                buf.clear()
            parts.append(p.strip())
        else:
            buf.extend(lines + [""])  # keep single blank to separate
    if buf:
        parts.append("\n".join(buf).strip())

    # Clean empties
    parts = [p for p in parts if p and p.strip()]
    return parts if parts else [text]


# ---------------------------
# Token-aware chunker
# ---------------------------

def _slide_windows(paragraphs: List[str], target: int, overlap: int, model: Optional[str]) -> List[dict]:
    """
    Slide a token window across paragraphs to create chunks respecting `target` and `overlap`.
    """
    chunks: List[dict] = []
    cur: List[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cur, cur_tokens
        if not cur:
            return
        text = "\n".join(cur).strip()
        tokens = count_tokens(text, model=model)
        chunks.append({"text": text, "tokens": tokens})
        cur = []
        cur_tokens = 0

    # Build chunks greedily up to `target`
    for para in paragraphs:
        t = count_tokens(para, model=model)
        if not cur:
            # Start new
            cur = [para]
            cur_tokens = t
            # Handle pathological long paragraph (> target): split by sentences
            if cur_tokens > target:
                flush()
            continue

        if cur_tokens + t <= target:
            cur.append(para)
            cur_tokens += t
        else:
            # Close current chunk
            flush()
            # Overlap: carry tail of previous chunk into the new one
            if overlap > 0 and chunks:
                tail = chunks[-1]["text"]
                # Take last ~overlap tokens by sentences (approximate)
                tail_sents = re.split(r"(?<=[\.\!\?])\s+", tail)
                acc = []
                acc_tokens = 0
                for s in reversed(tail_sents):
                    stoks = count_tokens(s, model=model)
                    if acc_tokens + stoks > overlap:
                        break
                    acc.insert(0, s)
                    acc_tokens += stoks
                if acc:
                    cur = [" ".join(acc), para]
                    cur_tokens = count_tokens(cur[0], model=model) + t
                else:
                    cur = [para]
                    cur_tokens = t
            else:
                cur = [para]
                cur_tokens = t

            # If the paragraph alone is too big, flush immediately
            if cur_tokens > target:
                flush()

    flush()
    return chunks


def chunk_text(
    text: str,
    *,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    model: Optional[str] = None,
    title_prefix: Optional[str] = None,
    hard_max_tokens: Optional[int] = 800,
) -> List[dict]:
    """
    Create token-aware chunks from raw text.

    Args:
      text           : input text (already normalized)
      target_tokens  : desired tokens per chunk (soft limit)
      overlap_tokens : tokens to overlap between adjacent chunks (context continuity)
      model          : tiktoken model name (if installed) for accurate counting
      title_prefix   : optional title/header prepended to each chunk to improve retrieval
      hard_max_tokens: absolute cap; chunks above this are forcibly split by sentences

    Returns:
      list of dicts: [{"text": "...", "position": 0, "tokens": 123, "title": "..."}]
    """
    if not text or not text.strip():
        return []

    # 1) Coarse segmentation by headings (if any), otherwise by paragraphs
    sections = split_by_headings(text)
    paragraphs: List[str] = []
    for sec in sections:
        paragraphs.extend(semantic_paragraphs(sec))

    # 2) Slide windows
    base_chunks = _slide_windows(paragraphs, target_tokens, overlap_tokens, model)

    # 3) Enforce hard max by splitting long chunks into sentences
    final_chunks: List[dict] = []
    for ch in base_chunks:
        if hard_max_tokens and ch["tokens"] > hard_max_tokens:
            sents = re.split(r"(?<=[\.\!\?])\s+", ch["text"])
            buf: List[str] = []
            buf_toks = 0
            for s in sents:
                st = count_tokens(s, model=model)
                if buf and buf_toks + st > int(hard_max_tokens * 0.9):
                    txt = " ".join(buf).strip()
                    final_chunks.append({"text": txt, "tokens": count_tokens(txt, model=model)})
                    buf = [s]
                    buf_toks = st
                else:
                    buf.append(s)
                    buf_toks += st
            if buf:
                txt = " ".join(buf).strip()
                final_chunks.append({"text": txt, "tokens": count_tokens(txt, model=model)})
        else:
            final_chunks.append(ch)

    # 4) Add position and optional title prefix
    out: List[dict] = []
    for i, ch in enumerate(final_chunks):
        txt = ch["text"]
        ttl = (title_prefix or "").strip() or None
        if ttl:
            # Title prefix pattern recommended by OpenAI for better retrieval: "Title\n\n<chunk>"
            full = f"{ttl}\n\n{txt}"
            toks = count_tokens(full, model=model)
            out.append({"text": full, "position": i, "tokens": toks, "title": ttl})
        else:
            out.append({"text": txt, "position": i, "tokens": ch["tokens"], "title": None})

    return out


# ---------------------------
# Table chunking
# ---------------------------

def chunk_table(df, *, max_rows_per_chunk: int = 30) -> List[dict]:
    """
    Chunk a pandas DataFrame into TSV blocks with up to `max_rows_per_chunk` rows each.
    Includes a small header line with column names.

    Returns:
      list of dicts: [{"text": "COL1\tCOL2\\n1\\t2\\n...", "position": 0, "rows": 30}]
    """
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        raise RuntimeError("pandas is required for chunk_table")

    if max_rows_per_chunk <= 0:
        max_rows_per_chunk = 30

    n = int(getattr(df, "shape", (0, 0))[0])
    if n == 0:
        return []

    cols = [str(c) for c in getattr(df, "columns", [])]
    header = "\t".join(cols)

    out: List[dict] = []
    pos = 0
    for start in range(0, n, max_rows_per_chunk):
        part = df.iloc[start : start + max_rows_per_chunk]
        # Convert to TSV rows
        lines = [header]
        for _, row in part.iterrows():
            vals = [str(row[c]) if row[c] is not None else "" for c in df.columns]
            lines.append("\t".join(vals))
        txt = "\n".join(lines).strip()
        out.append({"text": txt, "position": pos, "rows": len(part)})
        pos += 1

    return out
