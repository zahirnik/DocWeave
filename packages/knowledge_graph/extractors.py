# packages/knowledge_graph/extractors.py
"""
Knowledge Graph — lightweight claim extractors
==============================================

Goal
----
Turn parsed document text (paragraphs/chunks) into *candidate* ClaimPayloads
that a builder can persist into the KG. Keep this module tiny, fast, and easy
to reason about; avoid heavyweight NLP so it runs in workers and tests easily.

Design
------
- Pure-Python, dependency-light (stdlib + pydantic from schema.py).
- Heuristics only: regexes for metrics/targets/periods with conservative defaults.
- Confidence is a simple additive score; tune in tests (golden samples).
- Provenance is *first-class*: doc_id, page, chunk_id are kept in each claim.

Typical flow (used by builders.py)
----------------------------------
1) Builders split a document into *chunks* (you already do this for RAG).
2) For each chunk: call `extract_claims_from_text(...)`.
3) You get back a list[ClaimPayload]; the builder will:
   - create Claim nodes,
   - make Evidence nodes for (doc_id/page/chunk_id),
   - add edges: Claim --about--> Entity, Claim --quantifies--> Metric, Claim --supported_by--> Evidence.

Notes
-----
- `metric_aliases` lets you map your domain words to canonical metric names.
  Example: r"scope\\s*1" → "GHG Scope 1"
- If you need stronger NLP later, add an optional path that calls an LLM,
  but keep this heuristic path as a safe fallback.

"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from .schema import ClaimPayload, canonical_key


# -----------------------
# Normalisation utilities
# -----------------------

_whitespace_re = re.compile(r"\s+")
_sentence_split_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def normalise_text(text: str) -> str:
    """
    Basic normalisation:
    - collapse whitespace
    - strip leading/trailing spaces
    """
    return _whitespace_re.sub(" ", (text or "").strip())


def split_sentences(text: str) -> List[str]:
    """
    Tiny sentence splitter that works well enough for short business prose.
    We avoid heavy NLP to keep workers light and tests deterministic.
    """
    t = normalise_text(text)
    if not t:
        return []
    parts = _sentence_split_re.split(t)
    return [p.strip() for p in parts if p.strip()]


# -----------------------
# Metric detection (regex)
# -----------------------

# Default, conservative aliases; extend in configs as needed.
DEFAULT_METRIC_ALIASES: Dict[str, str] = {
    r"\bscope\s*1\b": "GHG Scope 1",
    r"\bscope\s*2\b": "GHG Scope 2",
    r"\bscope\s*3\b": "GHG Scope 3",
    r"\bghg\b|\bgreenhouse\s+gas(es)?\b": "GHG (general)",
    r"\bco2e?\b|\bCO2e\b": "GHG (general)",
    r"\bintensity\b": "Emissions Intensity",
    r"\brenewable(s| energy)?\b": "Renewable Energy",
    r"\brevenue\b": "Revenue",
    r"\bcapex\b|\bcapital\s+expenditure(s)?\b": "Capex",
    r"\bwater\s+(use|consumption|withdrawal)\b": "Water Use",
    r"\bwaste\b": "Waste",
    r"\bLTIFR\b|\blost\s+time\s+injury\b": "LTIFR",
}

def guess_metric_key(sentence: str, metric_aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Return a *canonical metric key* ('metric:ghg scope 1') if we detect a known metric.
    """
    text = sentence.lower()
    aliases = metric_aliases or DEFAULT_METRIC_ALIASES
    for pattern, canonical_name in aliases.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return canonical_key("metric", canonical_name)
    return None


# -----------------------
# Target / period / baseline / unit detection
# -----------------------

_percent_re = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*%\b")
_net_zero_re = re.compile(r"\bnet\s*zero\b", re.I)
_increase_re = re.compile(r"\b(increase|grow|rise|higher)\b", re.I)
_decrease_re = re.compile(r"\b(decrease|reduce|cut|lower|drop)\b", re.I)

# years: "by 2030", "to 2030", "between 2019 and 2030", "2019–2030"
_year_single_re = re.compile(r"\b(20\d{2}|19\d{2})\b")
_year_range_re = re.compile(
    r"\b(20\d{2}|19\d{2})\s*(?:–|-|to|through|until)\s*(20\d{2}|19\d{2})\b", re.I
)
_by_year_re = re.compile(r"\bby\s+(20\d{2}|19\d{2})\b", re.I)

# [NEW] baseline patterns (e.g., "vs 2019", "from a 2019 baseline", "relative to 2019")
_baseline_re = re.compile(
    r"\b(?:vs\.?|versus|against|relative\s+to|from)\s*(?:the\s*)?(?:\b(?:year|baseline)\b\s*)?(20\d{2}|19\d{2})\b",
    re.I,
)

# [NEW] simple unit patterns
_unit_tokens = [
    (re.compile(r"%"), "%"),
    (re.compile(r"\b(?:t|kt|mt)\s*co2e\b|\bco2e\b", re.I), "tCO2e"),
    (re.compile(r"\b(?:tonnes?|tons?)\b", re.I), "tonnes"),
    (re.compile(r"\b(?:k|m|bn)\b", re.I), "scalar"),  # generic scalar qualifier
    (re.compile(r"\b(?:kwh|mwh|gwh)\b", re.I), "Wh"),
    (re.compile(r"£|\bgbp\b|\bpounds?\b", re.I), "GBP"),
]

def guess_target(sentence: str) -> Optional[str]:
    """
    Extract a *target* phrase:
    - percentage target: "30%"
    - "net zero" target: "net zero"
    - directional targets: "reduce by 15%" → "15%"
    """
    # explicit percentage
    m = _percent_re.search(sentence)
    if m:
        return f"{m.group(1)}%"

    # net zero
    if _net_zero_re.search(sentence):
        return "net zero"

    # directional without explicit % (less precise)
    if _decrease_re.search(sentence):
        return "decrease"
    if _increase_re.search(sentence):
        return "increase"

    return None


def guess_period(sentence: str) -> Optional[str]:
    """
    Extract a *period* as a light, human-readable string:
    - "2019–2030"
    - "by 2030"
    - "2024"
    """
    # range first
    r = _year_range_re.search(sentence)
    if r:
        y1, y2 = r.group(1), r.group(2)
        return f"{y1}–{y2}"

    # "by YEAR"
    b = _by_year_re.search(sentence)
    if b:
        return f"by {b.group(1)}"

    # single year fallback
    s = _year_single_re.search(sentence)
    if s:
        return s.group(1)

    return None

# [NEW] baseline year detector
def guess_baseline_year(sentence: str) -> Optional[str]:
    b = _baseline_re.search(sentence)
    if b:
        return b.group(1)
    return None

# [NEW] unit detector (very light)
def guess_unit(sentence: str) -> Optional[str]:
    for rx, tag in _unit_tokens:
        if rx.search(sentence):
            return tag
    return None

# [NEW] direction detector for normalized text
def guess_direction(sentence: str) -> Optional[str]:
    if _decrease_re.search(sentence):
        return "reduction"
    if _increase_re.search(sentence):
        return "increase"
    if _net_zero_re.search(sentence):
        return "net_zero"
    return None


# -----------------------
# Confidence scoring
# -----------------------

def score_confidence(
    has_metric: bool,
    has_target: bool,
    has_period: bool,
    length_chars: int,
) -> float:
    """
    Simple additive confidence with conservative caps.

    Base 0.40
    +0.30 if we have a metric
    +0.20 if we have a target
    +0.10 if we have a period
    -0.10 if sentence is too short (< 24 chars)
    Clamp to [0.35, 0.95]
    """
    score = 0.40
    if has_metric:
        score += 0.30
    if has_target:
        score += 0.20
    if has_period:
        score += 0.10
    if length_chars < 24:
        score -= 0.10
    return max(0.35, min(0.95, score))


# -----------------------
# Public API
# -----------------------

def extract_claims_from_text(
    *,
    entity_key: str,
    text: str,
    doc_id: Optional[str] = None,
    page: Optional[int] = None,
    chunk_id: Optional[str] = None,
    metric_aliases: Optional[Dict[str, str]] = None,
) -> List[ClaimPayload]:
    """
    Extract ClaimPayloads from a chunk of text for a *known* entity.

    Parameters
    ----------
    entity_key:
        canonical entity key (e.g., "org:acme plc"). The builder knows this.
    text:
        raw paragraph/chunk text (already parsed/cleaned upstream).
    doc_id, page, chunk_id:
        provenance info for evidence linking (highly recommended).
    metric_aliases:
        optional dict[regex -> canonical metric name]; defaults cover ESG basics.

    Returns
    -------
    list[ClaimPayload]
        Each claim has metric/period/target (when detected), confidence, and provenance.

    Examples
    --------
    >>> claims = extract_claims_from_text(
    ...     entity_key="org:acme plc",
    ...     text="Acme will reduce Scope 1 emissions by 30% by 2030 versus 2019.",
    ...     doc_id="acme_2024.pdf", page=12, chunk_id="c-77"
    ... )
    >>> claims[0].metric_key
    'metric:ghg scope 1'
    >>> claims[0].target
    '30%'
    >>> claims[0].period
    'by 2030'
    """
    claims: List[ClaimPayload] = []
    for sent in split_sentences(text):
        if not sent:
            continue

        # Core signals
        mkey = guess_metric_key(sent, metric_aliases)
        tgt = guess_target(sent)
        per = guess_period(sent)

        # [NEW] extras for structured hashing and linking
        base = guess_baseline_year(sent)
        unit = guess_unit(sent)
        direction = guess_direction(sent)

        has_metric = mkey is not None
        has_target = tgt is not None
        has_period = per is not None

        # Skip if sentence is too generic and we found nothing actionable
        if not (has_metric or has_target or has_period):
            continue

        conf = score_confidence(has_metric, has_target, has_period, len(sent))

        # [NEW] build a compact normalized_text string (used in Claim.hash)
        norm_parts: List[str] = []
        if direction:
            norm_parts.append(f"{direction}")
        if tgt:
            norm_parts.append(f"value:{tgt}")
        if unit and unit != "%" and (tgt is None or "%" not in tgt):
            norm_parts.append(f"unit:{unit}")
        if base:
            norm_parts.append(f"baseline_year:{base}")
        if per:
            norm_parts.append(f"period:{per}")
        normalized_text = " ".join(norm_parts) if norm_parts else normalise_text(sent)

        # [NEW] create payload with extra fields when supported by the schema.
        # If your ClaimPayload does not define these fields, remove them here.
        payload_kwargs = dict(
            text=sent,
            entity_key=entity_key,
            metric_key=mkey,
            period=per,
            target=tgt,
            confidence=conf,
            doc_id=doc_id,
            page=page,
            chunk_id=chunk_id,
        )
        # Attempt to pass optional fields; if model rejects extras, your tests will reveal it.
        try:
            claim = ClaimPayload(
                **payload_kwargs,
                normalized_text=normalized_text,   # [NEW]
                baseline=base,                     # [NEW]
                unit=unit,                         # [NEW]
            )
        except Exception:
            # Fallback: construct without optional extras (keeps backward-compat)
            claim = ClaimPayload(**payload_kwargs)

        claims.append(claim)

    return claims


# -----------------------
# Optional: table helpers
# -----------------------

# We keep table extraction optional to avoid a hard pandas dependency here.
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def extract_claims_from_table(
    *,
    entity_key: str,
    table,  # DataFrame-like (pandas) or list-of-lists; we branch at runtime
    doc_id: Optional[str] = None,
    page: Optional[int] = None,
    chunk_id: Optional[str] = None,
    metric_headers: Iterable[str] = ("scope 1", "scope 2", "scope 3", "revenue", "capex", "intensity"),
) -> List[ClaimPayload]:
    """
    Heuristic table reader that turns rows into quantitative claims.

    Strategy
    --------
    - Look for a header row; find columns whose header matches a known *metric header*.
    - For each numeric cell in those columns, emit a claim with:
        text = "MetricName = <value> <unit?> for <row label?>"
        metric_key = canonical_key("metric", MetricName)
        period = try: year in another column or row label
        target = None (tables usually hold actuals, not targets)
        confidence ≈ 0.65 (we saw a metric + number; period may refine this)

    This is *very* conservative and designed to be deterministic and easy to test.

    Note: For real finance tables, you will want a stronger typed schema and
    units normalization. This keeps it simple for the MVP.

    """
    claims: List[ClaimPayload] = []

    if pd is not None and isinstance(table, pd.DataFrame):
        df = table.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        header_map: Dict[str, str] = {}
        for h in df.columns:
            for mh in metric_headers:
                if mh in h:
                    header_map[h] = mh  # map actual column -> canonical metric header

        # try to find a "year" column (very common)
        year_col = None
        for cand in ("year", "fiscal year", "fy", "period"):
            if cand in df.columns:
                year_col = cand
                break

        for _, row in df.iterrows():
            period = None
            if year_col:
                y = str(row.get(year_col) or "").strip()
                if re.fullmatch(r"(?:20|19)\d{2}", y):
                    period = y

            for col, metric_header in header_map.items():
                raw = row.get(col)
                if raw is None:
                    continue
                # numeric detection
                try:
                    val = float(str(raw).replace(",", ""))
                except Exception:
                    continue

                metric_name = _metric_header_to_name(metric_header)
                mkey = canonical_key("metric", metric_name)
                text = f"{metric_name} = {val}"

                conf = score_confidence(has_metric=True, has_target=False, has_period=bool(period), length_chars=len(text))

                # [NEW] include normalized_text/unit opportunistically
                try:
                    claim = ClaimPayload(
                        text=text,
                        entity_key=entity_key,
                        metric_key=mkey,
                        period=period,
                        target=None,
                        confidence=conf,
                        doc_id=doc_id,
                        page=page,
                        chunk_id=chunk_id,
                        normalized_text=f"value:{val} period:{period}" if period else f"value:{val}",
                        unit=None,
                        baseline=None,
                    )
                except Exception:
                    claim = ClaimPayload(
                        text=text,
                        entity_key=entity_key,
                        metric_key=mkey,
                        period=period,
                        target=None,
                        confidence=conf,
                        doc_id=doc_id,
                        page=page,
                        chunk_id=chunk_id,
                    )
                claims.append(claim)
        return claims

    # Fallback: list-of-lists (very simple)
    if isinstance(table, list) and table and isinstance(table[0], list):
        headers = [str(h).strip().lower() for h in table[0]]
        rows = table[1:]
        header_map = {}
        for idx, h in enumerate(headers):
            for mh in metric_headers:
                if mh in h:
                    header_map[idx] = mh
        try:
            year_idx = next(i for i, h in enumerate(headers) if h in ("year", "fiscal year", "fy", "period"))
        except StopIteration:
            year_idx = None

        for r in rows:
            period = None
            if year_idx is not None and year_idx < len(r):
                y = str(r[year_idx]).strip()
                if re.fullmatch(r"(?:20|19)\d{2}", y):
                    period = y
            for col_idx, metric_header in header_map.items():
                if col_idx >= len(r):
                    continue
                raw = r[col_idx]
                try:
                    val = float(str(raw).replace(",", ""))
                except Exception:
                    continue

                metric_name = _metric_header_to_name(metric_header)
                mkey = canonical_key("metric", metric_name)
                text = f"{metric_name} = {val}"

                conf = score_confidence(has_metric=True, has_target=False, has_period=bool(period), length_chars=len(text))

                try:
                    claim = ClaimPayload(
                        text=text,
                        entity_key=entity_key,
                        metric_key=mkey,
                        period=period,
                        target=None,
                        confidence=conf,
                        doc_id=doc_id,
                        page=page,
                        chunk_id=chunk_id,
                        normalized_text=f"value:{val} period:{period}" if period else f"value:{val}",
                        unit=None,
                        baseline=None,
                    )
                except Exception:
                    claim = ClaimPayload(
                        text=text,
                        entity_key=entity_key,
                        metric_key=mkey,
                        period=period,
                        target=None,
                        confidence=conf,
                        doc_id=doc_id,
                        page=page,
                        chunk_id=chunk_id,
                    )
                claims.append(claim)
        return claims

    # Unknown table type -> nothing
    return claims


# -----------------------
# tiny helpers
# -----------------------

def _metric_header_to_name(header: str) -> str:
    """
    Map a loose header token (like 'scope 1') to a canonical metric name.
    Extend this in configs if you need per-tenant names.
    """
    h = header.strip().lower()
    if "scope 1" in h:
        return "GHG Scope 1"
    if "scope 2" in h:
        return "GHG Scope 2"
    if "scope 3" in h:
        return "GHG Scope 3"
    if "intensity" in h:
        return "Emissions Intensity"
    if "revenue" in h:
        return "Revenue"
    if "capex" in h or "capital expenditure" in h:
        return "Capex"
    if "water" in h:
        return "Water Use"
    if "waste" in h:
        return "Waste"
    return header.title()
