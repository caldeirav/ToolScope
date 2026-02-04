from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import re
import unicodedata

from .types import CanonicalTool


ToolPreprocessor = Callable[[str], str]


def _truncate_chars(s: str, n: Optional[int]) -> str:
    if n is None:
        return s
    if n < 0:
        return s
    return s[:n]


@dataclass(frozen=True)
class ToolTextConfig:
    """
    Controls how a tool is converted to text for embedding.

    fields:
      Which parts of CanonicalTool to include in the embedding text.
      Supported: "name", "description", "schema", "tags"
      Default: ("name", "description")

    truncate:
      Max number of characters for the produced text (after assembling, before preprocessors).
      Default: 256

    preprocessors:
      Optional sequence of string->string transforms applied in order.
      Default: () (no preprocessing)
    """
    fields: Tuple[str, ...] = ("name", "description")
    truncate: Optional[int] = 256
    preprocessors: Tuple[ToolPreprocessor, ...] = ()

    def render(self, tool: CanonicalTool) -> str:
        parts: list[str] = []
        for f in self.fields:
            if f == "name":
                parts.append(tool.name or "")
            elif f == "description":
                parts.append(tool.description or "")
            elif f == "schema":
                parts.append(str(tool.input_schema or ""))
            elif f == "tags":
                parts.append(" ".join(tool.tags or ()))
            else:
                raise ValueError(f"Unsupported field in ToolTextConfig.fields: {f!r}")

        text = "\n".join(p for p in parts if p)
        text = _truncate_chars(text, self.truncate)

        for op in self.preprocessors:
            text = op(text)

        return text


# ---------- Optional preprocessors (user opt-in) ----------

def lowercase() -> ToolPreprocessor:
    return lambda s: s.lower()


def normalize_unicode(form: str = "NFKC") -> ToolPreprocessor:
    def _op(s: str) -> str:
        return unicodedata.normalize(form, s)
    return _op


def collapse_whitespace() -> ToolPreprocessor:
    ws = re.compile(r"\s+")
    return lambda s: ws.sub(" ", s).strip()


def strip_control_chars() -> ToolPreprocessor:
    def _op(s: str) -> str:
        return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"))
    return _op
