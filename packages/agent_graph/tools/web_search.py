# packages/agent_graph/tools/web_search.py
"""
Web search tool — tiny, opt-in wrapper (Tavily-first) with strict domain policy.

What this module provides
-------------------------
- WebSearchConfig: env-driven knobs (provider, API keys, freshness).
- WebSearch: small facade with a single method:
    search(query: str, top_k: int = 5, *, days: int | None = 365, allow_domains: list[str] | None = None) -> list[dict]

Return shape (normalized)
-------------------------
[
  {
    "id": "https://example.com/path",
    "title": "Page title",
    "url": "https://example.com/path",
    "snippet": "Short summary/extract …",
    "score": 0.0,                 # provider-specific confidence (scaled to 0..1 when available)
    "source": {"provider": "tavily", "raw": {...}},  # small provenance (raw trimmed)
    "published": "2024-06-10T08:30:00Z"              # when available
  },
  ...
]

Providers (opt-in)
------------------
- "tavily" (default when TAVILY_API_KEY is present)
    • pip install tavily-python
    • respects `days` (freshness) and `include_domains`/`exclude_domains`
    • best for research snippets

- "none"
    • no network calls; returns []

Safety & policy
---------------
- If `allow_domains` is provided, it is ANDed with the global policy allow-list.
- Every candidate URL is validated against `packages.agent_graph.policies.is_domain_allowed`.
- No content scraping here; this is **search results only**. Retrieval is done by your loaders.

Typical usage
-------------
from packages.agent_graph.tools.web_search import WebSearch

ws = WebSearch.from_env()
hits = ws.search("ACME 2024 annual report gross margin", top_k=5, days=365, allow_domains=["sec.gov"])
for h in hits:
    print(h["title"], h["url"])
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from packages.core.logging import get_logger
from packages.agent_graph.policies import is_domain_allowed

log = get_logger(__name__)


# ---------------------------
# Config
# ---------------------------

@dataclass
class WebSearchConfig:
    provider: str = "none"                 # "tavily" | "none"
    tavily_api_key: Optional[str] = None
    default_days: int = 365                # default freshness window
    max_top_k: int = 10                    # hard cap to keep costs bounded
    timeout_s: int = 20

    @staticmethod
    def from_env() -> "WebSearchConfig":
        api = os.getenv("TAVILY_API_KEY") or None
        prov = "tavily" if api else (os.getenv("WEB_SEARCH_PROVIDER") or "none")
        return WebSearchConfig(
            provider=prov.strip().lower(),
            tavily_api_key=api,
            default_days=int(os.getenv("WEB_SEARCH_DAYS", "365")),
            max_top_k=int(os.getenv("WEB_SEARCH_MAX_TOPK", "10")),
            timeout_s=int(os.getenv("WEB_SEARCH_TIMEOUT_S", "20")),
        )


# ---------------------------
# Facade
# ---------------------------

class WebSearch:
    """
    Opt-in web search (Tavily). Keeps a tiny, predictable API and strict allow-list.
    """

    def __init__(self, config: Optional[WebSearchConfig] = None):
        self.cfg = config or WebSearchConfig.from_env()
        self._tavily = None  # lazy

    @classmethod
    def from_env(cls) -> "WebSearch":
        return cls()

    # ---- public API ----

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        days: Optional[int] = None,
        allow_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a web search with strong safety filters. Returns normalized hits.
        """
        if not query or not query.strip():
            return []

        k = max(1, min(int(top_k), int(self.cfg.max_top_k)))
        freshness_days = int(days if days is not None else self.cfg.default_days)

        if self.cfg.provider == "tavily":
            hits = self._tavily_search(query, k, freshness_days, allow_domains=allow_domains)
        else:
            hits = []

        # Global policy filter (AND with user-provided allow_domains if any)
        filtered: List[Dict[str, Any]] = []
        for h in hits:
            url = h.get("url") or ""
            if not url:
                continue
            ok = is_domain_allowed(url)
            if not ok:
                continue
            if allow_domains:
                # enforce additional AND constraint
                ok_user = any(dom.lower() in url.lower() for dom in allow_domains)
                if not ok_user:
                    continue
            filtered.append(h)

        return filtered[:k]

    # ---- providers ----

    def _ensure_tavily(self):
        if self._tavily is not None:
            return
        if not self.cfg.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY not set; cannot use provider 'tavily'.")
        try:
            from tavily import TavilyClient  # type: ignore
        except Exception as e:
            raise RuntimeError("Tavily provider requires `tavily-python`. Install with: pip install tavily-python") from e
        self._tavily = TavilyClient(api_key=self.cfg.tavily_api_key)

    def _tavily_search(
        self,
        query: str,
        top_k: int,
        days: int,
        *,
        allow_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tavily web search → normalized hits.
        """
        self._ensure_tavily()
        assert self._tavily is not None

        include_domains = allow_domains or None  # Tavily can accept include_domains
        try:
            # Tavily "search" returns a dict with "results": [{title, url, content, score, published_date, ...}]
            # We ask for concise results (no full scraping here).
            resp = self._tavily.search(
                query=query,
                search_depth="basic",           # cheaper; "advanced" does more crawling
                max_results=int(top_k),
                include_domains=include_domains,
                time_range=f"{days}d" if days and days > 0 else None,
                include_images=False,
                include_answer=False,
                include_raw_content=False,
                # Tavily may support a timeout in recent SDKs; we guard with try/except
            )
        except TypeError:
            # Older SDK signature fallback
            resp = self._tavily.search(
                query=query,
                search_depth="basic",
                max_results=int(top_k),
                include_domains=include_domains,
                time_range=f"{days}d" if days and days > 0 else None,
            )
        except Exception as e:
            log.info("Tavily search failed: %s", e)
            return []

        results = (resp or {}).get("results") or []
        out: List[Dict[str, Any]] = []
        for r in results:
            url = str(r.get("url") or "")
            title = str(r.get("title") or "").strip()
            snippet = str(r.get("content") or "").strip()
            score = float(r.get("score") or 0.0)
            published = r.get("published_date") or r.get("published_time") or None

            # Clamp score to 0..1 if it's plausibly a percentage
            if score > 1.0:
                # Some SDKs return a 0..100 score
                score = min(1.0, score / 100.0)

            out.append(
                {
                    "id": url,
                    "title": title or url,
                    "url": url,
                    "snippet": snippet[:500],
                    "score": float(score),
                    "published": str(published) if published else None,
                    "source": {
                        "provider": "tavily",
                        # keep raw provenance small to avoid bloating logs
                        "raw": {k: v for k, v in r.items() if k in ("url", "title", "score", "published_date")},
                    },
                }
            )
        return out


__all__ = ["WebSearch", "WebSearchConfig"]
