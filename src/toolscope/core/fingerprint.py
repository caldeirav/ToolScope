import hashlib
import json
from typing import Any, Dict
from .types import ToolFingerprint


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint_tool(name: str, description: str, input_schema: Dict[str, Any], extra: Dict[str, Any] | None = None) -> ToolFingerprint:
    """
    Fingerprint must change if the tool meaningfully changes.
    We intentionally include name + description + schema.
    """
    base = {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "extra": extra or {},
    }
    h = hashlib.sha256(stable_json(base).encode("utf-8")).hexdigest()
    return ToolFingerprint(h)
