# tests/test_ingestion.py
"""
Ingestion pipeline smoke tests (tiny, offline-friendly).

What we verify
--------------
- Minimal loaders can read a few lightweight formats (TXT, JSONL).
- Normalisation removes hyphen-newlines and collapses whitespace.
- Chunker produces non-empty slices and respects max_chars/overlap roughly.
- EmbeddingClient (provider="auto") returns consistent vector shapes.

Notes
-----
- We *avoid* heavy deps (no OCR, no PDF parsing requirement).
- If optional repo loaders are missing, we transparently fall back to tiny helpers here.
- These are smoke tests, not exhaustive validations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

# ── Optional repo imports (graceful fallbacks)
try:
    from packages.ingestion.normalizers import normalize_text as repo_normalize  # type: ignore
except Exception:
    repo_normalize = None

try:
    from packages.retriever.chunking import chunk_text as repo_chunk  # type: ignore
except Exception:
    repo_chunk = None

from packages.retriever.embeddings import EmbeddingClient


# ──────────────────────────────────────────────────────────────────────────────
# Local fallbacks (kept identical to script fallbacks in spirit)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_text(s: str) -> str:
    """
    Collapse whitespace and remove hard hyphen line-breaks like '-\\n'.
    """
    return " ".join((s or "").replace("-\n", "").replace("\n", " ").split())


def _chunk_text(text: str, *, max_chars: int = 800, overlap: int = 120) -> List[str]:
    s = _normalize_text(text)
    if not s:
        return []
    out: List[str] = []
    step = max(1, max_chars - overlap)
    i, n = 0, len(s)
    while i < n:
        out.append(s[i : min(n, i + max_chars)])
        i += step
    return out


def normalize_text(s: str) -> str:
    return repo_normalize(s) if callable(repo_normalize) else _normalize_text(s)


def chunk_text(s: str, *, max_chars: int = 800, overlap: int = 120) -> List[str]:
    return repo_chunk(s, max_chars=max_chars, overlap=overlap) if callable(repo_chunk) else _chunk_text(
        s, max_chars=max_chars, overlap=overlap
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    base = tmp_path / "samples"
    base.mkdir(parents=True, exist_ok=True)

    # 1) TXT with hyphen-newline and extra whitespace to test normalizer
    txt = (
        "ACME PLC Annual Report 2024:\n"
        "Gross mar-\n"
        "gin improved to 38.2% year over\n"
        "year.   Operating    cash-flow also improved.\n"
    )
    (base / "acme.txt").write_text(txt, encoding="utf-8")

    # 2) JSONL with {text: "..."} rows
    jsonl_lines = [
        {"text": "Beta Corp Q3 2024 revenue was $1.26bn, up 14.5% YoY."},
        {"text": "Delta Inc net debt to EBITDA was 3.0x for FY2023."},
    ]
    with (base / "finance.jsonl").open("w", encoding="utf-8") as f:
        for row in jsonl_lines:
            f.write(json.dumps(row) + "\n")

    return base


# ──────────────────────────────────────────────────────────────────────────────
# Tests — loaders & normalisation
# ──────────────────────────────────────────────────────────────────────────────

def test_txt_load_and_normalize(sample_dir: Path):
    p = sample_dir / "acme.txt"
    raw = p.read_text(encoding="utf-8")
    norm = normalize_text(raw)
    assert "mar-\n" not in norm  # hyphen-newline removed
    assert "Gross margin" in norm  # words rejoined
    # whitespace collapsed
    assert "Operating cash-flow" in norm
    # sanity length
    assert len(norm) > 40


def test_jsonl_load_and_collect(sample_dir: Path):
    p = sample_dir / "finance.jsonl"
    texts: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("text") or d.get("content") or ""
            if isinstance(t, str) and t.strip():
                texts.append(normalize_text(t))
    assert len(texts) == 2
    assert any("revenue" in t.lower() for t in texts)
    assert any("net debt to ebitda" in t.lower() for t in texts)


# ──────────────────────────────────────────────────────────────────────────────
# Tests — chunking
# ──────────────────────────────────────────────────────────────────────────────

def test_chunking_overlap_and_bounds():
    long_text = " ".join(["Revenue increased 10% QoQ."] * 200)  # ~ many tokens
    chunks = chunk_text(long_text, max_chars=200, overlap=50)
    assert len(chunks) > 3
    # Chunks should not exceed max_chars
    assert all(len(c) <= 200 for c in chunks)
    # Overlap heuristic: consecutive chunks share a prefix/suffix segment
    if len(chunks) >= 2:
        assert chunks[0][-50:].split()[0:1] == chunks[1][:50].split()[0:1]


# ──────────────────────────────────────────────────────────────────────────────
# Tests — embeddings
# ──────────────────────────────────────────────────────────────────────────────

def test_embedding_shapes_and_consistency():
    embedder = EmbeddingClient(provider="auto", model_alias="mini")
    texts = [
        "ACME PLC gross margin was 38.2% in FY2024.",
        "Beta Corp revenue hit $1.26bn in Q3 2024.",
        "Delta Inc net debt/EBITDA was 3.0x.",
    ]
    vecs = embedder.embed(texts, batch_size=64)
    assert isinstance(vecs, list) and len(vecs) == len(texts)
    dim = len(vecs[0])
    assert dim > 8  # local fallback is small but > 8
    assert all(len(v) == dim for v in vecs)

    # Same input → identical vector (deterministic)
    v1 = embedder.embed([texts[0]])[0]
    v2 = embedder.embed([texts[0]])[0]
    assert len(v1) == dim and len(v2) == dim
    # Cosine of identical vectors should be ~1.0 (within float tolerance)
    import math
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1)) + 1e-12
    n2 = math.sqrt(sum(b * b for b in v2)) + 1e-12
    cos = dot / (n1 * n2)
    assert cos > 0.999


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end mini ingest (TXT + JSONL → chunk → embed)
# ──────────────────────────────────────────────────────────────────────────────

def test_end_to_end_ingest_chunk_embed(sample_dir: Path):
    # Gather texts from the sample dir (TXT + JSONL only, to keep it dependency-light)
    texts: List[str] = []

    # TXT
    raw = (sample_dir / "acme.txt").read_text(encoding="utf-8")
    texts.append(normalize_text(raw))

    # JSONL
    with (sample_dir / "finance.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = d.get("text") or ""
            if isinstance(t, str) and t.strip():
                texts.append(normalize_text(t))

    # Chunk all texts
    chunks: List[str] = []
    for t in texts:
        chunks.extend(chunk_text(t, max_chars=240, overlap=40))

    assert len(chunks) >= 3

    # Embed
    embedder = EmbeddingClient(provider="auto", model_alias="mini")
    vecs = embedder.embed(chunks, batch_size=64)

    # Basic shape checks
    dim = len(vecs[0])
    assert len(vecs) == len(chunks)
    assert all(len(v) == dim for v in vecs)
