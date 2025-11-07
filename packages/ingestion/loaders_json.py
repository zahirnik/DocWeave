# packages/ingestion/loaders_json.py
"""
JSON / JSONL loaders — tiny, explicit, schema-aware (optional).

What this module provides
-------------------------
- load_json(path: str, *, schema: dict | None = None) -> object
    Read a JSON file and (optionally) validate it against a JSON Schema.
    Returns the decoded Python object (dict/list/...).

- load_jsonl(path: str, *, schema: dict | None = None) -> list[object]
    Read a JSONL (one JSON object per line). Optionally validates **each line**
    against the schema. Returns a list of decoded objects.

Design goals
------------
- Keep behavior predictable and tutorial-clear.
- Validation is **optional**; if jsonschema is not installed, we give a friendly hint.
- Fail with clear error messages (line numbers for JSONL).

Dependencies (optional)
-----------------------
- jsonschema  (`pip install jsonschema`)  — only needed if you pass `schema=...`.

Examples
--------
from packages.ingestion.loaders_json import load_json, load_jsonl

obj = load_json("data/samples/company.json", schema={"type":"object","required":["name"]})
rows = load_jsonl("data/samples/records.jsonl")  # list of dicts

Schema notes
------------
- For JSONL, the same schema is applied to each line **independently**.
- If you need per-record *and* file-level validation, validate the returned list yourself.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Tuple


# ---------------------------
# Optional schema validation
# ---------------------------

def _have_jsonschema() -> bool:
    try:
        import jsonschema  # noqa: F401
        return True
    except Exception:
        return False


def _validate_schema(obj: Any, schema: dict) -> None:
    """
    Validate `obj` against JSON Schema if `jsonschema` is installed.
    Raises ValueError with a friendly message on validation errors.
    """
    try:
        import jsonschema
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Schema provided but 'jsonschema' package is not installed. "
            "Run: pip install jsonschema"
        ) from e

    try:
        jsonschema.validate(instance=obj, schema=schema)
    except jsonschema.ValidationError as e:
        path = ".".join([str(p) for p in e.path]) if getattr(e, "path", None) else ""
        where = f" at path '{path}'" if path else ""
        raise ValueError(f"JSON schema validation failed{where}: {e.message}") from e


# ---------------------------
# Public API
# ---------------------------

def load_json(path: str, *, schema: Optional[dict] = None) -> Any:
    """
    Read a JSON file from disk with UTF-8 decoding and optional schema validation.

    Raises:
      FileNotFoundError
      json.JSONDecodeError
      ValueError (on schema validation error)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if schema is not None:
        _validate_schema(data, schema)

    return data


def load_jsonl(path: str, *, schema: Optional[dict] = None, skip_blank: bool = True, comment_prefix: Optional[str] = None) -> List[Any]:
    """
    Read a JSONL file (one JSON object per line).

    Args:
      path           : file path
      schema         : optional JSON Schema to validate EACH line
      skip_blank     : ignore empty/whitespace-only lines (default True)
      comment_prefix : if set (e.g., "#"), lines starting with this are ignored

    Returns:
      list of decoded JSON objects (one per line)

    Raises:
      FileNotFoundError
      json.JSONDecodeError (with line number in the message)
      ValueError (schema validation error with line number)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out: List[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s and skip_blank:
                continue
            if comment_prefix and s.startswith(comment_prefix):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                # Re-raise but include the physical line number from the file
                raise json.JSONDecodeError(f"{e.msg} (at file line {i})", e.doc, e.pos) from e

            if schema is not None:
                try:
                    _validate_schema(obj, schema)
                except ValueError as e:
                    raise ValueError(f"Line {i}: {e}") from e

            out.append(obj)

    return out
