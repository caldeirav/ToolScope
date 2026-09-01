"""
BFCL Multiple AST scoring.

Follows official possible_answer semantics (any-of lists, optional args whose
first list value is ""). Tries the gorilla ``bfcl-eval`` checker when that
package is importable without shadowing this local package; otherwise uses a
faithful local implementation.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model import ParsedToolCall


def _normalize_value(v: Any) -> Any:
    if isinstance(v, str):
        v = v.strip()
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
        try:
            f = float(v)
            if f.is_integer():
                return int(f)
            return f
        except (ValueError, TypeError):
            pass
        lower = v.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        return v.lower()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _value_in(actual: Any, allowed: List[Any]) -> bool:
    na = _normalize_value(actual)
    for a in allowed:
        if a == "":
            continue
        if na == _normalize_value(a):
            return True
    return False


def local_ast_valid(
    pred: Optional[ParsedToolCall],
    possible_answer: Any,
) -> bool:
    """
    Grade a single predicted call against raw BFCL possible_answer.

    possible_answer is typically:
      [{func_name: {param: [val, ...] | ["", default] | scalar}}]
    """
    if pred is None or not pred.name:
        return False

    items: List[Dict] = []
    if isinstance(possible_answer, list):
        items = [x for x in possible_answer if isinstance(x, dict)]
    elif isinstance(possible_answer, dict):
        items = [possible_answer]
    if not items:
        return False

    matching = None
    for item in items:
        if pred.name in item:
            matching = item[pred.name]
            break
    if matching is None:
        return False
    if not isinstance(matching, dict):
        return True

    args = pred.arguments if isinstance(pred.arguments, dict) else {}
    for key, expected in matching.items():
        if isinstance(expected, list):
            if not expected or expected[0] == "":
                if key in args and args[key] not in ("", None):
                    allowed = [v for v in expected if v != ""]
                    if allowed and not _value_in(args[key], allowed):
                        return False
                continue
            if key not in args:
                return False
            if not _value_in(args[key], expected):
                return False
        else:
            if key not in args:
                return False
            if _normalize_value(args[key]) != _normalize_value(expected):
                return False
    return True


def _official_ast_valid(
    pred: ParsedToolCall,
    func_descriptions: List[Dict],
    possible_answer: Any,
    test_category: str = "multiple",
) -> Optional[bool]:
    """
    Call gorilla bfcl-eval's ast_checker if the *installed* package can be
    loaded under an alias (this repo's local ``bfcl_eval`` would otherwise win).
    """
    eval_dir = str(Path(__file__).resolve().parent.parent)
    saved_path = list(sys.path)
    saved_mods = {
        k: sys.modules[k]
        for k in list(sys.modules)
        if k == "bfcl_eval" or k.startswith("bfcl_eval.")
    }
    try:
        while eval_dir in sys.path:
            sys.path.remove(eval_dir)
        for k in list(saved_mods):
            sys.modules.pop(k, None)
        spec = importlib.util.find_spec("bfcl_eval.eval_checker.ast_eval.ast_checker")
        if spec is None:
            return None
        ast_mod = importlib.import_module("bfcl_eval.eval_checker.ast_eval.ast_checker")
        enums = importlib.import_module("bfcl_eval.constants.enums")
        language = getattr(enums, "Language")
        py = getattr(language, "PYTHON", None) or getattr(language, "Python", "Python")
        model_output = [{pred.name: pred.arguments}]
        pa = possible_answer
        if not isinstance(pa, list):
            pa = [pa]
        result = ast_mod.ast_checker(
            func_descriptions,
            model_output,
            pa,
            py,
            test_category,
            "default",
        )
        if isinstance(result, dict):
            return bool(result.get("valid"))
        return bool(result)
    except Exception:
        return None
    finally:
        sys.path[:] = saved_path
        for k, mod in saved_mods.items():
            sys.modules[k] = mod


def compute_ast_valid(
    pred: Optional[ParsedToolCall],
    possible_answer: Any,
    func_descriptions: Optional[List[Dict]] = None,
    test_category: str = "multiple",
) -> bool:
    """Headline argument-correctness metric for the paper protocol."""
    local = local_ast_valid(pred, possible_answer)
    if pred is None:
        return False
    if func_descriptions:
        official = _official_ast_valid(
            pred, func_descriptions, possible_answer, test_category=test_category
        )
        if official is not None:
            return official
    return local
