# packages/security/pii.py
"""
PII utilities — lightweight detection and masking helpers (UK/EU-aware).

What this module provides
-------------------------
- find_pii(text: str) -> list[dict]
    Return structured matches for common PII with spans and a normalized type.

- mask_pii(text: str, *, keep_last4_for_cards: bool = True) -> str
    Replace PII in free text with tokens like [EMAIL], [PHONE], [IBAN], [NI], etc.

- redact_obj(obj) -> obj
    Recursively mask any strings found inside dicts/lists/tuples.

Design goals
------------
- Tutorial-clear, dependency-free (regex only).
- Conservative patterns to avoid false positives.
- UK/EU flavour where easy: UK postcode, NI number, IBAN (generic + GB),
  simple UK phone forms, basic credit-card with Luhn check.

Notes & limitations
-------------------
- Names are NOT detected (high false-positive risk).
- Addresses are not fully parsed; only UK postcodes are recognized.
- Credentials detection beyond emails/keys should be handled by policy engine.

Examples
--------
from packages.security.pii import find_pii, mask_pii

text = "Email me at jane.doe@example.co.uk or +44 20 7946 0958. NI: QQ123456C."
print(find_pii(text))
print(mask_pii(text))
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

# ---------------------------
# Regex patterns (compiled)
# ---------------------------

# Email (conservative)
EMAIL_RE = re.compile(
    r"(?P<email>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,24})",
    re.IGNORECASE,
)

# International/UK-ish phone:
#  - +44 20 7946 0958, +447700900123, 020 7946 0958, 07700 900123
#  - Keep loose but bounded to 7–16 digits total
PHONE_RE = re.compile(
    r"(?P<phone>(?:\+?\d{1,3}[\s-]?)?(?:\(?0\d{1,4}\)?[\s-]?)?(?:\d[\s-]?){7,16})"
)

# UK National Insurance number (NI/NINO): AA 12 34 56 A (with constraints)
# Excludes invalid prefixes per gov.uk guidance (simplified set)
NI_RE = re.compile(
    r"(?P<ni>\b(?!BG)(?!GB)(?!NK)(?!KN)(?!TN)(?!NT)(?!ZZ)"
    r"[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b)",
    re.IGNORECASE,
)

# UK Postcode (simplified, good enough for most cases)
POSTCODE_RE = re.compile(
    r"(?P<postcode>\b"
    r"(GIR\s?0AA|"
    r"(?:[A-PR-UWYZ][0-9]{1,2}|"
    r"[A-PR-UWYZ][A-HK-Y][0-9]{1,2}|"
    r"[A-PR-UWYZ][0-9][A-HJKPSTUW]|"
    r"[A-PR-UWYZ][A-HK-Y][0-9][ABEHMNPRVWXY])"
    r"\s?[0-9][ABD-HJLNP-UW-Z]{2})\b)",
    re.IGNORECASE,
)

# IBAN (generic; 15–34 alphanumerics after 2-letter country and 2 digits)
IBAN_RE = re.compile(r"(?P<iban>\b[A-Z]{2}\d{2}[A-Z0-9]{11,32}\b)", re.IGNORECASE)

# Credit card candidate (13–19 digits possibly separated by spaces/dashes).
# We’ll run a Luhn check to confirm.
CARD_CANDIDATE_RE = re.compile(
    r"(?P<card>\b(?:\d[ -]?){13,19}\b)"
)

# IPv4 (basic; not strict 0–255 validation to keep simple)
IPV4_RE = re.compile(
    r"(?P<ip>\b\d{1,3}(?:\.\d{1,3}){3}\b)"
)

# API key-ish tokens (very conservative; complementary to policy regex)
APIKEY_RE = re.compile(
    r"(?P<apikey>(?i)\b(?:api[_\- ]?key|secret|token)\b\s*[:=]\s*[A-Za-z0-9_\-]{12,})"
)

# ---------------------------
# Helpers
# ---------------------------

def _luhn_ok(digits_only: str) -> bool:
    """Return True if the numeric string passes Luhn checksum."""
    try:
        s = [int(c) for c in digits_only if c.isdigit()]
    except Exception:
        return False
    if len(s) < 13:
        return False
    checksum = 0
    parity = (len(s) - 2) % 2
    for i, n in enumerate(s[:-1]):
        if i % 2 == parity:
            n = n * 2
            if n > 9:
                n -= 9
        checksum += n
    checksum = (checksum + s[-1]) % 10
    return checksum == 0


def _norm_phone(s: str) -> str:
    """Strip non-digits to gauge plausibility of a phone number."""
    digits = re.sub(r"\D", "", s)
    return digits


def _iter_matches(text: str) -> Iterable[Tuple[str, str, Tuple[int, int]]]:
    """
    Yield (type, value, (start, end)) tuples for each PII type found.
    Applies extra validation where needed (cards, phones).
    """
    # Email
    for m in EMAIL_RE.finditer(text):
        yield ("email", m.group("email"), m.span())

    # Phone (apply simple plausibility: 7–16 digits)
    for m in PHONE_RE.finditer(text):
        raw = m.group("phone")
        digits = _norm_phone(raw)
        if 7 <= len(digits) <= 16:
            yield ("phone", raw, m.span())

    # NI number
    for m in NI_RE.finditer(text):
        yield ("ni", m.group("ni"), m.span())

    # UK postcode
    for m in POSTCODE_RE.finditer(text):
        yield ("postcode", m.group("postcode"), m.span())

    # IBAN (leave deep checksum to caller; pattern is often sufficient)
    for m in IBAN_RE.finditer(text):
        yield ("iban", m.group("iban"), m.span())

    # Credit card: candidate + Luhn
    for m in CARD_CANDIDATE_RE.finditer(text):
        raw = m.group("card")
        digits = re.sub(r"\D", "", raw)
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            yield ("card", raw, m.span())

    # IPv4
    for m in IPV4_RE.finditer(text):
        yield ("ip", m.group("ip"), m.span())

    # API key-ish
    for m in APIKEY_RE.finditer(text):
        yield ("apikey", m.group("apikey"), m.span())


# ---------------------------
# Public API
# ---------------------------

def find_pii(text: str) -> List[Dict[str, object]]:
    """
    Return a list of matches:
      [{"type": "email", "value": "...", "start": 10, "end": 25}, ...]
    """
    out: List[Dict[str, object]] = []
    for typ, val, (a, b) in _iter_matches(text or ""):
        out.append({"type": typ, "value": val, "start": int(a), "end": int(b)})
    return out


def mask_pii(text: str, *, keep_last4_for_cards: bool = True) -> str:
    """
    Replace PII substrings in `text` with category tokens.

    Categories:
      [EMAIL], [PHONE], [NI], [POSTCODE], [IBAN], [CARD], [IP], [APIKEY]

    For credit cards, keeps last 4 digits if `keep_last4_for_cards` is True.
    """
    s = text or ""
    matches = list(_iter_matches(s))
    if not matches:
        return s

    # Replace from right to left to keep spans valid
    out = s
    for typ, val, (a, b) in sorted(matches, key=lambda t: t[2][0], reverse=True):
        token = f"[{typ.upper()}]"
        if typ == "card" and keep_last4_for_cards:
            last4 = re.sub(r"\D", "", val)[-4:]
            token = f"[CARD••{last4}]"
        out = out[:a] + token + out[b:]
    return out


def redact_obj(obj: Any) -> Any:
    """
    Recursively mask strings in dict/list/tuple structures.
    Leaves non-strings unchanged.
    """
    if isinstance(obj, str):
        return mask_pii(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(redact_obj(x) for x in obj)
    return obj


__all__ = ["find_pii", "mask_pii", "redact_obj"]
