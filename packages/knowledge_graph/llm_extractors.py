# packages/knowledge_graph/llm_extractors.py
"""
Knowledge Graph — optional LLM extractors
=========================================

Purpose
-------
Stronger claim extraction when heuristics aren't enough. This module calls an
LLM to extract claims/metrics/targets/periods from a chunk of text and returns
validated `ClaimPayload` objects with provenance.

Design
------
- Dependency-light by default. If OpenAI (or LangChain) is not installed, we
  raise a friendly RuntimeError explaining how to enable LLM extraction.
- Deterministic shape: we ask for *structured JSON* and validate with Pydantic.
- Conservative defaults: we clamp confidence into [0,1] and keep targets/periods
  as short strings (exact normalisation can happen later if needed).

When to use
-----------
- Keep `extractors.py` (regex) as your fast baseline.
- Use this module for trickier sentences or as a “second pass” when heuristics
  return no claims.

Environment
-----------
- Set `OPENAI_API_KEY` for the OpenAI path.
- You can also pass a client explicitly if you manage credentials elsewhere.

Examples
--------
>>> from packages.knowledge_graph.llm_extractors import llm_extract_claims
>>> claims = llm_extract_claims(
...     entity_key="org:acme plc",
...     text="Acme targets a 30% reduction in Scope 1 emissions by 2030 versus a 2019 baseline.",
...     doc_id="acme_2024.pdf", page=12, chunk_id="c-77",
... )
>>> claims[0].target
'30%'

Notes
-----
- We avoid tight coupling to specific SDK versions:
  * Try `from openai import OpenAI` first (new SDK), else fallback to legacy.
- If you prefer LangChain’s structured output, see `llm_extract_claims_langchain`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .schema import ClaimPayload


# -----------------------
# Pydantic response schema
# -----------------------

class _LLMClaim(BaseModel):
    text: str = Field(..., description="The exact sentence/phrase representing the claim.")
    metric: Optional[str] = Field(None, description="Canonical metric name, e.g., 'GHG Scope 1' or 'Revenue'.")
    target: Optional[str] = Field(None, description="Target expression if present (e.g., '30%' or 'net zero').")
    period: Optional[str] = Field(None, description="Time expression (e.g., 'by 2030', '2019–2030', '2024').")
    confidence: float = Field(0.70, description="Model-estimated extraction confidence in [0,1].")

    @field_validator("confidence")
    @classmethod
    def _clamp_conf(cls, v: float) -> float:
        try:
            v = float(v)
        except Exception:
            return 0.70
        return max(0.0, min(1.0, v))


class _LLMResponse(BaseModel):
    claims: List[_LLMClaim] = Field(default_factory=list)


# -----------------------
# Prompt
# -----------------------

_SYSTEM = """You extract *factual claims* about an entity from business text.
Return only JSON matching the provided schema. Be conservative:
- If you cannot find a clear metric/target/period, leave that field null.
- Use short strings for target/period (e.g., "30%", "by 2030", "2019–2030").
- confidence is in [0,1]."""

_USER_TMPL = """Entity key: {entity_key}
Text (single chunk):
\"\"\"{text}\"\"\"

Return JSON:
{{
  "claims": [
    {{
      "text": "...",
      "metric": "... or null",
      "target": "... or null",
      "period": "... or null",
      "confidence": 0.0
    }}
  ]
}}"""


# -----------------------
# OpenAI — minimal client shims
# -----------------------

def _get_openai_client(explicit_client: Any = None):
    """
    Return an OpenAI-like client. We support:
      - New SDK: from openai import OpenAI → OpenAI(api_key=...)
      - Legacy SDK: import openai; openai.ChatCompletion.create(...)
    If neither is available or no API key, raise a clear error.
    """
    if explicit_client is not None:
        return ("modern", explicit_client)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Set it to enable LLM extraction, or use heuristics only.")

    # Try modern SDK first
    try:
        from openai import OpenAI  # type: ignore
        return ("modern", OpenAI(api_key=api_key))
    except Exception:
        pass

    # Fallback to legacy
    try:
        import openai  # type: ignore
        openai.api_key = api_key
        return ("legacy", openai)
    except Exception as exc:
        raise RuntimeError(
            "OpenAI SDK not installed. Install with: `pip install openai` "
            "or pass an explicit client to llm_extract_claims(..., client=...)."
        ) from exc


# -----------------------
# Public API — OpenAI
# -----------------------

def llm_extract_claims(
    *,
    entity_key: str,
    text: str,
    doc_id: Optional[str] = None,
    page: Optional[int] = None,
    chunk_id: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    client: Any = None,
) -> List[ClaimPayload]:
    """
    Extract claims via OpenAI chat, returning validated ClaimPayloads.

    Parameters
    ----------
    entity_key : canonical entity key (e.g., "org:acme plc")
    text       : chunk text
    doc_id/page/chunk_id : provenance for evidence linkage
    model      : OpenAI model name (default 'gpt-4o-mini')
    temperature: 0.0 for most deterministic outputs
    client     : optional OpenAI client (modern or legacy). If None, we build one.

    Returns
    -------
    list[ClaimPayload]
    """
    mode, cli = _get_openai_client(client)

    if mode == "modern":
        # New SDK (responses API under the hood exposed via chat.completions)
        try:
            resp = cli.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _USER_TMPL.format(entity_key=entity_key, text=text)},
                ],
            )
            content = resp.choices[0].message.content or "{}"
        except Exception as exc:
            raise RuntimeError(f"OpenAI chat call failed: {exc}") from exc
    else:
        # Legacy SDK
        try:
            resp = cli.ChatCompletion.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _USER_TMPL.format(entity_key=entity_key, text=text)},
                ],
                functions=None,
            )
            content = resp["choices"][0]["message"]["content"] or "{}"
        except Exception as exc:
            raise RuntimeError(f"OpenAI chat call failed: {exc}") from exc

    # Parse & validate JSON
    try:
        raw = json.loads(content)
    except Exception:
        # Some models return text with code fences. Try to salvage.
        content = content.strip().strip("`").strip()
        if content.startswith("json"):
            content = content[4:].strip()
        raw = json.loads(content)

    parsed = _LLMResponse.model_validate(raw)

    # Convert to ClaimPayloads with provenance
    out: List[ClaimPayload] = []
    for c in parsed.claims:
        out.append(
            ClaimPayload(
                text=c.text.strip(),
                entity_key=entity_key,
                metric_key=(f"metric:{c.metric.strip().lower()}" if c.metric else None),
                period=(c.period.strip() if c.period else None),
                target=(c.target.strip() if c.target else None),
                confidence=float(c.confidence),
                doc_id=doc_id,
                page=page,
                chunk_id=chunk_id,
            )
        )
    return out


# -----------------------
# Optional: LangChain structured output
# -----------------------

def llm_extract_claims_langchain(
    *,
    entity_key: str,
    text: str,
    doc_id: Optional[str] = None,
    page: Optional[int] = None,
    chunk_id: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> List[ClaimPayload]:
    """
    Alternative implementation using LangChain's structured output helpers.
    Requires `pip install langchain langchain-openai`.

    This is kept short on purpose; prefer `llm_extract_claims` unless you
    specifically want LangChain in this layer.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        from langchain_core.pydantic_v1 import BaseModel as LCBaseModel, Field as LCField  # type: ignore
        from langchain_core.output_parsers import PydanticOutputParser  # type: ignore
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "LangChain not installed. Install with: `pip install langchain langchain-openai` "
            "or use llm_extract_claims(...)"
        ) from exc

    class LCClaim(LCBaseModel):
        text: str
        metric: Optional[str] = LCField(default=None)
        target: Optional[str] = LCField(default=None)
        period: Optional[str] = LCField(default=None)
        confidence: float = 0.70

    class LCResp(LCBaseModel):
        claims: List[LCClaim] = []

    parser = PydanticOutputParser(pydantic_object=LCResp)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            ("user", _USER_TMPL + "\n\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    llm = ChatOpenAI(model=model, temperature=temperature)
    chain = prompt | llm | parser
    out = chain.invoke({"entity_key": entity_key, "text": text})

    claims: List[ClaimPayload] = []
    for c in out.claims:
        claims.append(
            ClaimPayload(
                text=c.text.strip(),
                entity_key=entity_key,
                metric_key=(f"metric:{c.metric.strip().lower()}" if c.metric else None),
                period=(c.period.strip() if c.period else None),
                target=(c.target.strip() if c.target else None),
                confidence=float(c.confidence),
                doc_id=doc_id,
                page=page,
                chunk_id=chunk_id,
            )
        )
    return claims


__all__ = ["llm_extract_claims", "llm_extract_claims_langchain"]
