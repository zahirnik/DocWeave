# packages/agent_graph/tools/kg_context.py
"""
Optional Knowledge-Graph (KG) context — tiny, file-backed helper to enrich prompts.

What this module provides
-------------------------
A minimal, dependency-light facility to keep simple finance facts (triples) that can
augment retrieval context. Think of it as a tiny "entity memory" you can seed from
CSV/JSONL and query by entity strings.

Class
-----
KGContext(root_dir="./data/kg", file_name="facts.jsonl")
  - add_facts(facts: list[dict]) -> int
      facts are dicts with keys:
        {"subject": str, "predicate": str, "object": str,
         "source": Optional[str], "confidence": Optional[float]}
      Duplicates (same s/p/o) are de-duplicated on load.

  - lookup(text: str, top_k: int = 10) -> list[dict]
      Simple case-insensitive substring match over subject/object; returns triples.

  - related_entities(entity: str, top_k: int = 10) -> list[dict]
      Neighbour summary: [{"entity": "...", "count": 3, "predicates": ["owns","acquired"]}, ...]

  - enrich_prompt(query: str, max_items: int = 6) -> str
      Produce a compact bullet list of facts to append to an LLM prompt.

  - load_csv(path: str) -> int
      Load triples from CSV with columns: subject,predicate,object,(optional)source,confidence.

Design goals
------------
- Tutorial-clear; zero heavy deps (no RDF libraries).
- File format is JSONL (one triple per line), easy to inspect and version.
- Writes are atomic; safe on crashes.
- This is **optional**. Your graph does not depend on it. Use only if helpful.

Example
-------
kg = KGContext()
kg.add_facts([
    {"subject":"ACME PLC","predicate":"ticker","object":"ACM.L","source":"manual","confidence":1.0},
    {"subject":"ACME PLC","predicate":"sector","object":"Consumer Staples","source":"manual"},
])
print(kg.enrich_prompt("Compare ACME PLC and Beta Corp margins"))
"""

from __future__ import annotations

import csv
import json
import os
import uuid
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from packages.core.logging import get_logger

log = get_logger(__name__)


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    source: Optional[str] = None
    confidence: Optional[float] = None
    updated_at: str = ""

    def key(self) -> Tuple[str, str, str]:
        return (self.subject.strip().lower(), self.predicate.strip().lower(), self.object.strip().lower())

    def to_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }
        if self.source:
            d["source"] = self.source
        if self.confidence is not None:
            d["confidence"] = float(self.confidence)
        if self.updated_at:
            d["updated_at"] = self.updated_at
        return d

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "Triple":
        return Triple(
            subject=str(d.get("subject") or "").strip(),
            predicate=str(d.get("predicate") or "").strip(),
            object=str(d.get("object") or "").strip(),
            source=(str(d.get("source")) if d.get("source") is not None else None),
            confidence=(float(d["confidence"]) if d.get("confidence") is not None else None),
            updated_at=str(d.get("updated_at") or ""),
        )


class KGContext:
    """
    Minimal triple store backed by a JSONL file. Not a database; intended for small fact sets.
    """

    def __init__(self, root_dir: str = "./data/kg", file_name: str = "facts.jsonl"):
        self.root_dir = root_dir
        self.file_name = file_name
        _ensure_dir(self.root_dir)
        self._path = os.path.join(self.root_dir, self.file_name)
        self._triples: Dict[Tuple[str, str, str], Triple] = {}
        self._load()

    # ------------- load/save -------------

    def _load(self) -> None:
        self._triples.clear()
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        t = Triple.from_dict(d)
                        if t.subject and t.predicate and t.object:
                            self._triples[t.key()] = t
                    except Exception:
                        continue
        except Exception as e:
            log.info("KG load failed (%s); starting empty.", e)

    def _save(self) -> None:
        lines = []
        for t in self._triples.values():
            lines.append(json.dumps(t.to_dict(), ensure_ascii=False))
        text = "\n".join(lines) + ("\n" if lines else "")
        _atomic_write(self._path, text)

    # ------------- public API -------------

    def add_facts(self, facts: Iterable[Dict[str, object]]) -> int:
        """
        Upsert triples from dicts. Returns number of upserts (new or changed).
        """
        up = 0
        now = _utc_iso()
        for d in facts or []:
            t = Triple.from_dict(d)
            if not (t.subject and t.predicate and t.object):
                continue
            t = Triple(
                subject=t.subject,
                predicate=t.predicate,
                object=t.object,
                source=t.source,
                confidence=t.confidence,
                updated_at=now,
            )
            k = t.key()
            old = self._triples.get(k)
            if not old or old.to_dict() != t.to_dict():
                self._triples[k] = t
                up += 1
        if up:
            self._save()
        return up

    def load_csv(self, path: str) -> int:
        """
        Load triples from a CSV with headers: subject,predicate,object,(optional)source,confidence.
        Returns number of upserts.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        facts: List[Dict[str, object]] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                s = (row.get("subject") or "").strip()
                p = (row.get("predicate") or "").strip()
                o = (row.get("object") or "").strip()
                if not (s and p and o):
                    continue
                src = (row.get("source") or "").strip() or None
                conf = row.get("confidence")
                facts.append(
                    {
                        "subject": s,
                        "predicate": p,
                        "object": o,
                        "source": src,
                        "confidence": float(conf) if conf not in (None, "",) else None,
                    }
                )
        return self.add_facts(facts)

    def lookup(self, text: str, top_k: int = 10) -> List[Dict[str, object]]:
        """
        Return triples where subject/object contains the text (case-insensitive).
        """
        if not text:
            return []
        q = text.lower().strip()
        hits: List[Tuple[int, Triple]] = []
        for t in self._triples.values():
            s = t.subject.lower()
            o = t.object.lower()
            # crude scoring: exact match gets 2, substring gets 1
            score = 0
            if q == s or q == o:
                score = 2
            elif q in s or q in o:
                score = 1
            if score > 0:
                hits.append((score, t))
        hits.sort(key=lambda x: (x[0], x[1].updated_at), reverse=True)
        out: List[Dict[str, object]] = []
        for _, t in hits[: max(1, int(top_k))]:
            d = t.to_dict()
            d["score"] = _
            out.append(d)
        return out

    def related_entities(self, entity: str, top_k: int = 10) -> List[Dict[str, object]]:
        """
        Neighbour summary for an entity (by subject/object adjacency).
        """
        if not entity:
            return []
        q = entity.lower().strip()
        counts: Dict[str, Dict[str, object]] = {}
        for t in self._triples.values():
            s = t.subject.lower().strip()
            o = t.object.lower().strip()
            if q == s:
                k = t.object
                rec = counts.setdefault(k, {"entity": k, "count": 0, "predicates": set()})
                rec["count"] = int(rec["count"]) + 1
                rec["predicates"].add(t.predicate)
            elif q == o:
                k = t.subject
                rec = counts.setdefault(k, {"entity": k, "count": 0, "predicates": set()})
                rec["count"] = int(rec["count"]) + 1
                rec["predicates"].add(t.predicate)
        items = list(counts.values())
        for it in items:
            it["predicates"] = sorted(list(it["predicates"]))  # type: ignore[assignment]
        items.sort(key=lambda x: (x["count"], len(x["predicates"])), reverse=True)
        return items[: max(1, int(top_k))]

    def enrich_prompt(self, query: str, max_items: int = 6) -> str:
        """
        Produce a small bullet list of facts relevant to terms in the query.
        Uses a simple token split; for serious usage, integrate NER.
        """
        toks = [t.strip(",. ").lower() for t in (query or "").split() if len(t) >= 3]
        seen_keys = set()
        facts: List[str] = []

        # Try each token as a lookup
        for t in toks:
            for hit in self.lookup(t, top_k=3):
                key = (hit["subject"], hit["predicate"], hit["object"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                src = f" (src: {hit.get('source')})" if hit.get("source") else ""
                conf = hit.get("confidence")
                conf_s = f" [conf={conf:.2f}]" if isinstance(conf, (float, int)) else ""
                facts.append(f"• {hit['subject']} — {hit['predicate']} — {hit['object']}{conf_s}{src}")
                if len(facts) >= max_items:
                    break
            if len(facts) >= max_items:
                break

        return "\n".join(facts) if facts else ""

    # ------------- debug / maintenance -------------

    def stats(self) -> Dict[str, object]:
        n = len(self._triples)
        by_pred: Dict[str, int] = {}
        for t in self._triples.values():
            by_pred[t.predicate] = by_pred.get(t.predicate, 0) + 1
        return {"count": n, "by_predicate": dict(sorted(by_pred.items(), key=lambda kv: kv[1], reverse=True))}

    def clear(self) -> None:
        self._triples.clear()
        if os.path.exists(self._path):
            try:
                os.remove(self._path)
            except Exception:
                pass
