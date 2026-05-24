"""HMAC-SHA256 anonymization of customer_ids before any LLM exposure.

The salt is generated once and stored at cache/salt.txt; subsequent runs read
the same salt so anonymized IDs are stable across reruns.
"""

from __future__ import annotations
import hashlib
import hmac
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SALT_PATH = ROOT / "cache" / "salt.txt"


def _load_or_create_salt() -> bytes:
    SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SALT_PATH.exists():
        return SALT_PATH.read_text().strip().encode()
    salt = secrets.token_hex(32)
    SALT_PATH.write_text(salt)
    return salt.encode()


_SALT = _load_or_create_salt()


def anonymize(customer_id: str) -> str:
    return hmac.new(_SALT, customer_id.encode(), hashlib.sha256).hexdigest()[:16]
