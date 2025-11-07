# packages/ingestion/normalizers.py
"""
Text normalizers — tiny, explicit, and safe for finance documents.

What this module provides
-------------------------
- normalize_text(s: str, **flags) -> str
    One high-level function to clean OCR/PDF/HTML-extracted text with sensible defaults.

- Individual helpers (pure functions, easy to test):
    * to_unicode_nfkc(s)
    * dehyphenate_linebreaks(s)
    * collapse_blank_lines(s, max_blank=1)
    * trim_trailing_spaces(s)
    * collapse_inner_whitespace(s)
    * strip_page_artifacts(s, header=None, footer=None, page_hint_re=None)
    * fix_bullets_and_dashes(s)
    * normalize_quotes(s)
    * ascii_safe(s, keep="£€$%")  # optional for tabular exports

Design goals
------------
- Keep it **tutorial-clear** and dependency-light (stdlib only).
- Deterministic transforms: same input → same output (no random heuristics).
- Heuristics tuned for annual/quarterly reports, investor decks, term sheets.
- Each helper is small and composable; you can reuse them in other pipelines.

Usage
-----
from packages.ingestion.normalizers import normalize_text

clean = normalize_text(raw_string, dehyphenate=True, collapse_ws=True)

Testing tips
-----------
- Run helpers individually in unit tests to verify exact behavior.
- Keep a tiny "golden" input/output pair for regression tests.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional


# ---------------------------
# Core helpers
# ---------------------------

def to_unicode_nfkc(s: str) -> str:
    """
    Normalize Unicode to NFKC form:
    - Canonical composition (é vs e + ´)
    - Compatibility (ﬀ → ff, full-width → ASCII)
    """
    try:
        return unicodedata.normalize("NFKC", s)
    except Exception:
        return s


def trim_trailing_spaces(s: str) -> str:
    """
    Remove trailing spaces/tabs on every line, preserve line structure.
    """
    return "\n".join(line.rstrip(" \t") for line in s.splitlines())


def collapse_blank_lines(s: str, max_blank: int = 1) -> str:
    """
    Collapse runs of blank lines to at most `max_blank` (default 1).
    """
    if max_blank < 0:
        max_blank = 0
    # Replace \r\n to \n first to standardize
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Find groups of 2+ newlines and limit them
    pattern = re.compile(r"\n{2,}")
    replacement = "\n" * (max_blank + 1) if max_blank >= 1 else "\n"
    return pattern.sub(replacement, s)


def collapse_inner_whitespace(s: str) -> str:
    """
    Collapse runs of 2+ spaces/tabs inside lines to a single space.
    (Does not touch newlines.)
    """
    return "\n".join(re.sub(r"[ \t]{2,}", " ", line) for line in s.splitlines())


def dehyphenate_linebreaks(s: str) -> str:
    """
    Join words broken across line breaks with hyphens from PDF extraction/OCR.

    Example:
        "inter-\nnational" → "international"
        "multi-\n line"   → "multi line"  (keeps a space if next line starts with a space)

    Heuristic:
        - If a line ends with a letter + hyphen and next line begins with a letter,
          remove the hyphen and the newline.
        - Preserve capitalization exactly.
    """
    lines = s.splitlines()
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if re.search(r"[A-Za-z]\-$", cur) and re.match(r"^[A-Za-z]", nxt):
                # Join: remove trailing '-' and newline
                out.append(cur[:-1] + nxt)
                i += 2
                continue
        out.append(cur)
        i += 1
    return "\n".join(out)


def normalize_quotes(s: str) -> str:
    """
    Normalize curly quotes/apostrophes to straight quotes where safe.
    Keeps special finance symbols intact (£, €, $).
    """
    repl = {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‛": "'",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def fix_bullets_and_dashes(s: str) -> str:
    """
    Normalize lists and dashes:
      •, ‣, ▪, – → '-'
      em-dash (—) → " - "
    Avoids double conversions (idempotent).
    """
    s = s.replace("•", "- ").replace("‣", "- ").replace("▪", "- ")
    s = s.replace("–", "-")
    # Surround em-dash with spaces to avoid word concatenation
    s = re.sub(r"\s*—\s*", " - ", s)
    # De-duplicate repeated spaces introduced
    s = re.sub(r"[ ]{2,}", " ", s)
    return s


def strip_page_artifacts(
    s: str,
    *,
    header: Optional[str] = None,
    footer: Optional[str] = None,
    page_hint_re: Optional[str] = r"^\s*Page\s+\d+(\s*/\s*\d+)?\s*$",
) -> str:
    """
    Remove simple repeating headers/footers and page number lines.

    Args:
      header       : exact header line to strip when it appears
      footer       : exact footer line to strip when it appears
      page_hint_re : regex applied to a *full line* (e.g., "Page 1 of 10")

    Note:
      This is intentionally conservative. For complex artifacts, handle in the loader.
    """
    lines = s.splitlines()
    out = []
    rex = re.compile(page_hint_re) if page_hint_re else None
    for ln in lines:
        if header and ln.strip() == header.strip():
            continue
        if footer and ln.strip() == footer.strip():
            continue
        if rex and rex.match(ln.strip()):
            continue
        out.append(ln)
    return "\n".join(out)


def ascii_safe(s: str, keep: str = "£€$%") -> str:
    """
    Convert text to ASCII where possible, preserving selected finance symbols.

    - Keeps characters in `keep`.
    - Replaces other non-ASCII chars with spaces (then collapses spaces).
    """
    buf = []
    for ch in s:
        if ch in keep:
            buf.append(ch)
        else:
            code = ord(ch)
            if 32 <= code <= 126:  # printable ASCII
                buf.append(ch)
            elif ch in ("\n", "\t"):
                buf.append(ch)
            else:
                buf.append(" ")
    out = "".join(buf)
    # Collapse spaces introduced by replacements (not across newlines)
    out = "\n".join(re.sub(r" {2,}", " ", ln) for ln in out.splitlines())
    return out


# ---------------------------
# High-level pipeline
# ---------------------------

def normalize_text(
    s: str,
    *,
    unicode_nfkc: bool = True,
    strip_headers_footers: bool = True,
    header: Optional[str] = None,
    footer: Optional[str] = None,
    dehyphenate: bool = True,
    fix_lists: bool = True,
    normalize_quotes_flag: bool = True,
    collapse_ws: bool = True,
    collapse_blank: int = 1,
    ascii_only: bool = False,
    ascii_keep: str = "£€$%",
) -> str:
    """
    Opinionated normalization pipeline suitable for chunking/embedding.

    Steps (toggle via flags):
      1) Unicode NFKC (compatibility folding)                 [unicode_nfkc]
      2) Trim trailing spaces per line                        [always]
      3) Strip simple headers/footers and page number lines   [strip_headers_footers]
      4) De-hyphenate broken words across line breaks         [dehyphenate]
      5) Normalize bullets/dashes                             [fix_lists]
      6) Normalize quotes/apostrophes                         [normalize_quotes_flag]
      7) Collapse inner whitespace (runs of spaces/tabs)      [collapse_ws]
      8) Collapse blank lines (max = collapse_blank)          [collapse_blank]
      9) ASCII-safe pass (optional)                           [ascii_only]

    Args:
      s: raw text
      header/footer: exact lines to remove if present (optional)
      ascii_only: if True, non-ASCII characters are stripped except `ascii_keep`

    Returns:
      Cleaned text string.
    """
    if not isinstance(s, str):
        s = str(s)

    if unicode_nfkc:
        s = to_unicode_nfkc(s)

    s = trim_trailing_spaces(s)

    if strip_headers_footers:
        s = strip_page_artifacts(s, header=header, footer=footer)

    if dehyphenate:
        s = dehyphenate_linebreaks(s)

    if fix_lists:
        s = fix_bullets_and_dashes(s)

    if normalize_quotes_flag:
        s = normalize_quotes(s)

    if collapse_ws:
        s = collapse_inner_whitespace(s)

    s = collapse_blank_lines(s, max_blank=max(0, int(collapse_blank)))

    if ascii_only:
        s = ascii_safe(s, keep=ascii_keep)

    return s


# ---------------------------
# Convenience: line filters
# ---------------------------

def filter_lines(s: str, *, drop_if_re: Optional[str] = None, keep_if_re: Optional[str] = None) -> str:
    """
    Keep or drop lines based on regex matches.

    - If `keep_if_re` is provided, only lines matching it are kept.
    - Else if `drop_if_re` is provided, lines matching it are removed.
    - If neither provided, returns the input unchanged.

    Useful for pruning watermark lines or keeping only SECTION headings.
    """
    if not drop_if_re and not keep_if_re:
        return s
    keep_re = re.compile(keep_if_re) if keep_if_re else None
    drop_re = re.compile(drop_if_re) if drop_if_re else None

    lines = s.splitlines()
    out = []
    for ln in lines:
        if keep_re:
            if keep_re.search(ln):
                out.append(ln)
        elif drop_re and drop_re.search(ln):
            continue
        else:
            out.append(ln)
    return "\n".join(out)


# ---------------------------
# Backwards-compatibility alias
# ---------------------------

def clean_text(text: str, *args, **kwargs) -> str:  # [NEW]
    """
    Backwards-compatible alias used by worker/tasks and API routes.

    Delegates to `normalize_text`. Any keyword flags accepted by `normalize_text`
    can be passed through (e.g., dehyphenate=False, collapse_ws=False).
    """
    return normalize_text(text, **kwargs)
