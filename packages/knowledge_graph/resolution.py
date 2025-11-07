# packages/knowledge_graph/resolution.py
"""
resolution.py
Entity & metric resolution utilities (multi-tenant) for the Knowledge Graph.

Features
--------
- Canonicalization of names/keys
- Alias registry (in-memory, optional JSON file)
- Exact match on name/alt_names
- Full-Text Search (Neo4j FTS) fallback via fts_entity_name
- Metric resolution by exact key/name with fuzzy fallback (difflib)
- Content-hash helper for stable claim IDs

Dependencies
------------
- neo4j (Python driver)
- Optional: a JSON alias file at configs/entity_aliases.json

Usage
-----
from packages.knowledge_graph.resolution import Neo4jResolver
r = Neo4jResolver.from_env()
hits = r.resolve_entity("J Sainsbury plc", tenant_id="public")
"""

from __future__ import annotations

import json
import os
import re
import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from neo4j import GraphDatabase, Driver
except Exception as e:  # pragma: no cover
    Driver = Any  # type: ignore

try:
    from rapidfuzz.fuzz import ratio as fuzz_ratio  # optional
except Exception:  # fallback to stdlib
    from difflib import SequenceMatcher

    def fuzz_ratio(a: str, b: str) -> float:  # type: ignore
        return SequenceMatcher(None, a, b).ratio() * 100.0

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join([c for c in text if not unicodedata.combining(c)])

def canonicalize_text(text: str) -> str:
    """
    Lowercase, strip accents, collapse whitespace and punctuation commonly found
    in org/person names to improve matching stability.
    """
    t = text.strip().lower()
    t = _strip_accents(t)
    # unify punctuation and whitespace
    t = re.sub(r"[\s\-_\./]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def stable_hash(*parts: str, length: int = 40) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return h[:length]

@dataclass
class ResolvedEntity:
    id: str
    name: str
    score: float
    method: str  # "alias", "exact", "alt", "fts"
    raw: Optional[Dict[str, Any]] = None

@dataclass
class ResolvedMetric:
    id: str
    key: str
    score: float
    method: str  # "alias", "exact_key", "exact_name", "fuzzy"

class AliasRegistry:
    """
    In-memory alias registry with optional JSON bootstrap.
    JSON format (list or dict):
      { "org:sainsburys": ["j sainsbury plc", "sainsbury's", "sainsburys plc"] }
    or
      [ {"id":"org:sainsburys", "aliases":["j sainsbury plc", ...]} ]
    All aliases are stored in canonicalized form.
    """
    def __init__(self, mapping: Optional[Dict[str, List[str]]] = None) -> None:
        self.mapping: Dict[str, List[str]] = {}
        self.inverse: Dict[str, str] = {}  # alias -> canonical id
        if mapping:
            self._load(mapping)

    def _load(self, mapping: Dict[str, List[str]]) -> None:
        for cid, aliases in mapping.items():
            cset: List[str] = []
            for a in aliases:
                ca = canonicalize_text(a)
                if ca and ca not in cset:
                    cset.append(ca)
                    # first wins; later duplicates ignore
                    self.inverse.setdefault(ca, cid)
            self.mapping[cid] = cset

    @classmethod
    def from_json_file(cls, path: str) -> "AliasRegistry":
        if not os.path.exists(path):
            return cls({})
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            mapping = {k: list(v) for k, v in data.items()}
        elif isinstance(data, list):
            mapping = {}
            for row in data:
                cid = row.get("id")
                aliases = row.get("aliases", [])
                if cid:
                    mapping[cid] = aliases
        else:
            mapping = {}
        return cls(mapping)

    def lookup(self, text: str) -> Optional[str]:
        return self.inverse.get(canonicalize_text(text))

# ──────────────────────────────────────────────────────────────────────────────
# Resolver
# ──────────────────────────────────────────────────────────────────────────────

class Neo4jResolver:
    def __init__(
        self,
        driver: Driver,
        entity_aliases_path: str = "configs/entity_aliases.json",
        metric_aliases_path: str = "configs/metric_aliases.json",
    ) -> None:
        self.driver = driver
        self.entity_aliases = AliasRegistry.from_json_file(entity_aliases_path)
        self.metric_aliases = AliasRegistry.from_json_file(metric_aliases_path)

    # Factory using env
    @classmethod
    def from_env(cls) -> "Neo4jResolver":
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER")
        pwd = os.environ.get("NEO4J_PASSWORD")
        if not uri or not user or not pwd:
            raise RuntimeError("Missing NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD")
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        return cls(driver)

    # ── Entity resolution ─────────────────────────────────────────────────────

    def resolve_entity(
        self,
        name_or_alias: str,
        tenant_id: str,
        limit: int = 5,
        min_score: float = 0.55,
    ) -> List[ResolvedEntity]:
        """
        Returns best-matching entities for the given name/alias:
          1) Alias registry
          2) Exact match on name
          3) Exact match on alt_names
          4) FTS (db.index.fulltext.queryNodes('fts_entity_name', q))
        """
        q_raw = name_or_alias.strip()
        if not q_raw:
            return []
        q = canonicalize_text(q_raw)

        # 1) Alias registry
        alias_id = self.entity_aliases.lookup(q)
        results: List[ResolvedEntity] = []
        if alias_id:
            row = self._get_entity_by_id(tenant_id, alias_id)
            if row:
                results.append(
                    ResolvedEntity(
                        id=row["id"],
                        name=row.get("name", alias_id),
                        score=1.0,
                        method="alias",
                        raw=row,
                    )
                )
                return results  # alias is considered definitive

        # 2) Exact name
        row = self._get_entity_by_exact_name(tenant_id, q)
        if row:
            results.append(
                ResolvedEntity(
                    id=row["id"],
                    name=row.get("name", row["id"]),
                    score=1.0,
                    method="exact",
                    raw=row,
                )
            )
            return results

        # 3) Exact alt name
        row = self._get_entity_by_exact_alt(tenant_id, q)
        if row:
            results.append(
                ResolvedEntity(
                    id=row["id"],
                    name=row.get("name", row["id"]),
                    score=0.95,
                    method="alt",
                    raw=row,
                )
            )
            return results

        # 4) Full-text search fallback
        fts = self._fts_entities(tenant_id, q, limit=limit)
        results.extend(fts)

        # Filter by min_score
        return [r for r in results if r.score >= min_score]

    # ── Metric resolution ─────────────────────────────────────────────────────

    def resolve_metric(
        self,
        key_or_name: str,
        tenant_id: str,
        min_score: float = 0.6,
    ) -> List[ResolvedMetric]:
        q_raw = key_or_name.strip()
        if not q_raw:
            return []
        q = canonicalize_text(q_raw)

        # Alias registry first
        alias_id = self.metric_aliases.lookup(q)
        if alias_id:
            row = self._get_metric_by_id(tenant_id, alias_id)
            if row:
                return [ResolvedMetric(id=row["id"], key=row.get("key", row["id"]), score=1.0, method="alias")]

        # Exact by key
        row = self._get_metric_by_key(tenant_id, q)
        if row:
            return [ResolvedMetric(id=row["id"], key=row.get("key", row["id"]), score=1.0, method="exact_key")]

        # Exact by name (if provided)
        row = self._get_metric_by_name(tenant_id, q)
        if row:
            return [ResolvedMetric(id=row["id"], key=row.get("key", row["id"]), score=0.95, method="exact_name")]

        # Fuzzy fallback across existing metric keys/names
        all_metrics = self._list_metric_keys_and_names(tenant_id)
        scored: List[Tuple[float, Dict[str, str]]] = []
        for m in all_metrics:
            cand = canonicalize_text(m.get("key") or m.get("name") or "")
            if not cand:
                continue
            s = fuzz_ratio(q, cand) / 100.0
            scored.append((s, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[ResolvedMetric] = []
        for s, m in scored[:5]:
            if s >= min_score:
                out.append(ResolvedMetric(id=m["id"], key=m.get("key", m["id"]), score=s, method="fuzzy"))
        return out

    # ── Claim content-hash helper ─────────────────────────────────────────────

    def claim_content_hash(
        self,
        tenant_id: str,
        entity_id: str,
        metric_id: str,
        normalized_text: str,
        period: Optional[str] = None,
        baseline: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> str:
        parts = [
            canonicalize_text(tenant_id),
            entity_id,
            metric_id,
            canonicalize_text(normalized_text),
            canonicalize_text(period or ""),
            canonicalize_text(baseline or ""),
            canonicalize_text(unit or ""),
        ]
        return stable_hash(*parts, length=40)

    # ──────────────────────────────────────────────────────────────────
    # Neo4j queries
    # ──────────────────────────────────────────────────────────────────

    def _get_entity_by_id(self, tenant_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
        cypher = (
            "MATCH (e:Entity {tenant_id:$tenant, id:$id}) "
            "RETURN e.id AS id, e.name AS name, e.alt_names AS alt_names LIMIT 1"
        )
        with self.driver.session() as s:
            rec = s.run(cypher, tenant=tenant_id, id=entity_id).single()
            return rec.data() if rec else None

    def _get_entity_by_exact_name(self, tenant_id: str, canonical_name: str) -> Optional[Dict[str, Any]]:
        # store both raw and canonical for resilience
        cypher = """
        MATCH (e:Entity {tenant_id:$tenant})
        WHERE toLower(e.name) = $q
           OR toLower(replace(e.name, "'", '')) = $q
        RETURN e.id AS id, e.name AS name, e.alt_names AS alt_names
        LIMIT 1
        """
        with self.driver.session() as s:
            rec = s.run(cypher, tenant=tenant_id, q=canonical_name).single()
            return rec.data() if rec else None

    def _get_entity_by_exact_alt(self, tenant_id: str, canonical_name: str) -> Optional[Dict[str, Any]]:
        cypher = (
            "MATCH (e:Entity {tenant_id:$tenant}) "
            "WHERE any(a IN coalesce(e.alt_names, []) WHERE toLower(a) = $q) "
            "RETURN e.id AS id, e.name AS name, e.alt_names AS alt_names LIMIT 1"
        )
        with self.driver.session() as s:
            rec = s.run(cypher, tenant=tenant_id, q=canonical_name).single()
            return rec.data() if rec else None

    def _fts_entities(self, tenant_id: str, query: str, limit: int = 5) -> List[ResolvedEntity]:
        cypher = (
            "CALL db.index.fulltext.queryNodes('fts_entity_name', $q) "
            "YIELD node, score "
            "WHERE node.tenant_id = $tenant "
            "RETURN node.id AS id, node.name AS name, score "
            "ORDER BY score DESC LIMIT $limit"
        )
        with self.driver.session() as s:
            rows = s.run(cypher, q=query, tenant=tenant_id, limit=limit).data()
        # Neo4j FTS scores are not bounded to 1.0; normalize roughly
        if not rows:
            return []
        max_score = max(r["score"] for r in rows) or 1.0
        out: List[ResolvedEntity] = []
        for r in rows:
            norm = float(r["score"]) / float(max_score)
            out.append(ResolvedEntity(id=r["id"], name=r.get("name") or r["id"], score=norm, method="fts"))
        return out

    def _get_metric_by_id(self, tenant_id: str, metric_id: str) -> Optional[Dict[str, Any]]:
        cypher = (
            "MATCH (m:Metric {tenant_id:$tenant, id:$id}) "
            "RETURN m.id AS id, m.key AS key, m.name AS name LIMIT 1"
        )
        with self.driver.session() as s:
            rec = s.run(cypher, tenant=tenant_id, id=metric_id).single()
            return rec.data() if rec else None

    def _get_metric_by_key(self, tenant_id: str, key: str) -> Optional[Dict[str, Any]]:
        cypher = (
            "MATCH (m:Metric {tenant_id:$tenant}) "
            "WHERE toLower(m.key) = $q "
            "RETURN m.id AS id, m.key AS key, m.name AS name LIMIT 1"
        )
        with self.driver.session() as s:
            rec = s.run(cypher, tenant=tenant_id, q=key).single()
            return rec.data() if rec else None

    def _get_metric_by_name(self, tenant_id: str, name: str) -> Optional[Dict[str, Any]]:
        cypher = (
            "MATCH (m:Metric {tenant_id:$tenant}) "
            "WHERE toLower(m.name) = $q "
            "RETURN m.id AS id, m.key AS key, m.name AS name LIMIT 1"
        )
        with self.driver.session() as s:
            rec = s.run(cypher, tenant=tenant_id, q=name).single()
            return rec.data() if rec else None

    def _list_metric_keys_and_names(self, tenant_id: str) -> List[Dict[str, str]]:
        cypher = (
            "MATCH (m:Metric {tenant_id:$tenant}) "
            "RETURN m.id AS id, m.key AS key, m.name AS name"
        )
        with self.driver.session() as s:
            return s.run(cypher, tenant=tenant_id).data()
