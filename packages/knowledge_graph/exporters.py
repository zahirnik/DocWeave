# packages/knowledge_graph/exporters.py
"""
Export helpers for the Knowledge Graph:
- to_json(nodes, edges, tenant_id, meta=None) -> dict
- to_graphml(nodes, edges, graph_id="kg", directed=True, pretty=False) -> bytes

GraphML is built with xml.etree.ElementTree and uses attribute dicts
(e.g., {"for": "node"}) to avoid Python reserved keywords like `for=`.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional, Sequence
from xml.etree import ElementTree as ET

from .schema import Node, Edge


# -----------------------
# Label helpers (robust fallbacks)
# -----------------------

_ACRONYM_FIXES = {
    "ghg": "GHG",
    "co2e": "CO2e",
    "ltifr": "LTIFR",
}

_WORD_SEP = re.compile(r"\s+")

def _titleish(s: str) -> str:
    """Title case with acronym fixes."""
    if not s:
        return s
    parts = _WORD_SEP.split(s.strip())
    out = []
    for w in parts:
        lw = w.lower()
        if lw in _ACRONYM_FIXES:
            out.append(_ACRONYM_FIXES[lw])
        else:
            out.append(w.capitalize())
    return " ".join(out)

def _pretty_metric_from_key(key: str) -> str:
    # key form: "metric:ghg scope 2"
    name = key.split(":", 1)[-1] if ":" in key else key
    return _titleish(name)

def _pretty_entity_from_key(key: str) -> str:
    # key form: "org:j sainsbury plc"
    name = key.split(":", 1)[-1] if ":" in key else key
    # Keep PLC/LLC/Inc uppercase if present
    name = _titleish(name)
    name = re.sub(r"\bPlc\b", "PLC", name)
    name = re.sub(r"\bLlc\b", "LLC", name)
    name = re.sub(r"\bInc\b", "Inc.", name)
    return name

def _sentence_case(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    return s

def _claim_text_from_key(key: str) -> str:
    # key form: "claim:<entity_key>|<truncated claim text>"
    if "|" in key:
        text = key.split("|", 1)[1]
    else:
        text = key
    # De-canonicalize common lowercasing from canonical_key
    text = _sentence_case(text)
    return text

def _safe_label(n: Node) -> str:
    """
    Robust label selection for UIs expecting node['label'].
    Priority:
      1) n.label if present and non-empty
      2) props-derived label (by type)
      3) parsed from key (by type)
      4) fallback to key
    """
    if getattr(n, "label", None):
        if isinstance(n.label, str) and n.label.strip():
            return n.label

    p = n.props or {}
    t = n.type.value

    if t == "claim":
        # Prefer full text from props
        txt = p.get("text")
        if txt and isinstance(txt, str) and txt.strip():
            return _sentence_case(txt.strip())
        return _claim_text_from_key(n.key)

    if t == "metric":
        nm = p.get("name")
        if nm and isinstance(nm, str) and nm.strip():
            return _titleish(nm)
        return _pretty_metric_from_key(n.key)

    if t == "entity":
        nm = p.get("name")
        if nm and isinstance(nm, str) and nm.strip():
            # Preserve proper company suffix capitalization
            return _pretty_entity_from_key(f"org:{nm}")
        return _pretty_entity_from_key(n.key)

    if t == "evidence":
        cit = p.get("citation")
        if cit and isinstance(cit, str) and cit.strip():
            return cit.strip()
        # Keep evidence key as last resort (locator)
        return n.key

    # Default fallback
    return n.key


# -----------------------
# JSON export
# -----------------------

def to_json(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    *,
    tenant_id: Optional[str] = None,
    meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Convert nodes/edges into a simple JSON-serialisable payload.
    """
    t_id = tenant_id or (nodes[0].tenant_id if nodes else (edges[0].tenant_id if edges else "t0"))

    def _node(n: Node) -> Dict[str, object]:
        return {
            "id": str(n.id),
            "tenant_id": n.tenant_id,
            "type": n.type.value,
            "key": n.key,
            "label": _safe_label(n),  # always non-empty string fallback
            "props": n.props or {},
        }

    def _edge(e: Edge) -> Dict[str, object]:
        return {
            "id": str(e.id),
            "tenant_id": e.tenant_id,
            "type": e.type.value,
            "src_id": str(e.src_id),
            "dst_id": str(e.dst_id),
            "props": e.props or {},
        }

    payload: Dict[str, object] = {
        "meta": {
            "tenant_id": t_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "nodes": [_node(n) for n in nodes],
        "edges": [_edge(e) for e in edges],
    }
    if meta:
        payload["meta"] = {**payload["meta"], **meta}
    return payload


# -----------------------
# GraphML export
# -----------------------

def to_graphml(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    *,
    graph_id: str = "kg",
    directed: bool = True,
    pretty: bool = False,
) -> bytes:
    """
    Convert the KG into GraphML bytes suitable for tools like Gephi.
    """
    NS = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", NS)

    def _el(tag: str, attrib: Optional[Dict[str, str]] = None, text: Optional[str] = None) -> ET.Element:
        e = ET.Element(f"{{{NS}}}{tag}", attrib or {})
        if text is not None:
            e.text = text
        return e

    def _sub(parent: ET.Element, tag: str, attrib: Optional[Dict[str, str]] = None, text: Optional[str] = None) -> ET.Element:
        e = ET.SubElement(parent, f"{{{NS}}}{tag}", attrib or {})
        if text is not None:
            e.text = text
        return e

    root = _el("graphml")

    # Key declarations (node/edge attributes)
    _sub(root, "key", {"id": "n_type", "for": "node", "attr.name": "type", "attr.type": "string"})
    _sub(root, "key", {"id": "n_key",  "for": "node", "attr.name": "key",  "attr.type": "string"})
    _sub(root, "key", {"id": "n_props","for": "node", "attr.name": "props","attr.type": "string"})
    _sub(root, "key", {"id": "e_type", "for": "edge", "attr.name": "type", "attr.type": "string"})
    _sub(root, "key", {"id": "e_props","for": "edge", "attr.name": "props","attr.type": "string"})

    g = _sub(root, "graph", {"id": graph_id, "edgedefault": "directed" if directed else "undirected"})

    # Nodes
    for n in nodes:
        n_el = _sub(g, "node", {"id": str(n.id)})
        _sub(n_el, "data", {"key": "n_type"}, n.type.value)
        _sub(n_el, "data", {"key": "n_key"}, n.key)
        props_json = json.dumps(n.props or {}, ensure_ascii=False)
        _sub(n_el, "data", {"key": "n_props"}, props_json)

    # Edges
    for e in edges:
        e_el = _sub(g, "edge", {"id": str(e.id), "source": str(e.src_id), "target": str(e.dst_id)})
        _sub(e_el, "data", {"key": "e_type"}, e.type.value)
        props_json = json.dumps(e.props or {}, ensure_ascii=False)
        _sub(e_el, "data", {"key": "e_props"}, props_json)

    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    # Normalise XML declaration to double quotes for test consistency
    if xml.startswith(b"<?xml version='1.0' encoding='utf-8'?>"):
        xml = xml.replace(b"<?xml version='1.0' encoding='utf-8'?>", b'<?xml version="1.0" encoding="utf-8"?>', 1)

    if not pretty:
        return xml

    # Optional pretty-print
    try:
        import xml.dom.minidom as minidom
        dom = minidom.parseString(xml)
        xml_pretty = dom.toprettyxml(indent="  ", encoding="utf-8")
        if xml_pretty.startswith(b"<?xml version='1.0' encoding='utf-8'?>"):
            xml_pretty = xml_pretty.replace(
                b"<?xml version='1.0' encoding='utf-8'?>",
                b'<?xml version="1.0" encoding="utf-8"?>',
                1,
            )
        return xml_pretty
    except Exception:
        return xml


__all__ = ["to_json", "to_graphml"]
