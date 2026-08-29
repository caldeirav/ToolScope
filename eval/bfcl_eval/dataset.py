"""
BFCL dataset loading, caching, and tool pool construction.

Downloads BFCL v3 data from GitHub, caches locally, converts to OpenAI
tool format, and builds the shared distractor pool used across all instances.
"""

import ast
import json
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BFCL_BASE_URL = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main"
    "/berkeley-function-call-leaderboard/bfcl_eval/data"
)

CATEGORY_FILES = {
    "simple": "BFCL_v4_simple_python",
    "multiple": "BFCL_v4_multiple",
    "parallel": "BFCL_v4_parallel",
    "parallel_multiple": "BFCL_v4_parallel_multiple",
}


@dataclass
class BFCLEntry:
    id: str
    messages: List[Dict]      # [{"role": "user", "content": "..."}]
    functions: List[Dict]     # OpenAI-format tool dicts
    ground_truth: List[Dict]  # [{"func_name": {"arg": val, ...}}, ...]
    functions_bfcl: List[Dict] = field(default_factory=list)
    possible_answer: Any = None


def _download(url: str, dest: Path) -> None:
    label = dest.name
    if dest.parent.name == "possible_answer":
        label = f"possible_answer/{dest.name}"
    print(f"  Downloading {label} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            dest.write_bytes(r.read())
    except Exception as e:
        raise RuntimeError(
            f"Could not download {url}\n"
            f"  → {e}\n"
            f"  Check your network connection or pre-place the file at {dest}"
        ) from e


def _load_jsonl_or_array(path: Path) -> List[Dict]:
    """Parse a file that is either a JSON array or JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return [d for d in data if isinstance(d, dict)]
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    entries.append(obj)
            except json.JSONDecodeError:
                pass
    return entries


def _fix_params(params: Any) -> Dict:
    """
    Convert BFCL function parameters to OpenAI-compatible JSON Schema.

    BFCL v4 uses "type": "dict" for object parameters; OpenAI expects "type": "object".
    """
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    result = dict(params)
    if result.get("type") == "dict":
        result["type"] = "object"
    return result


def _bfcl_fn_to_openai(fn: Dict) -> Optional[Dict]:
    """Convert a BFCL function definition to OpenAI tool format."""
    name = fn.get("name", "")
    if not isinstance(name, str) or not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": fn.get("description") or "",
            "parameters": _fix_params(fn.get("parameters")),
        },
    }


def _normalise_gt_args(raw_args: Any) -> Dict:
    """
    Flatten BFCL v4 list-based argument values to plain scalars.

    BFCL v4 encodes ground-truth args as lists:
      [value]          -> required arg; expected value is `value`
      ["", default]    -> optional arg with a default; skip in evaluation
      [val1, val2]     -> multiple acceptable values; use the first

    Plain scalars (v3 or inline answers) are passed through unchanged.
    """
    if not isinstance(raw_args, dict):
        return {}
    result: Dict = {}
    for key, val in raw_args.items():
        if isinstance(val, list):
            if not val or val[0] == "":
                continue  # optional arg -- omit from expected args
            result[key] = val[0]
        else:
            result[key] = val
    return result


def _parse_ground_truth(gt_raw: Any) -> List[Dict]:
    """
    Parse a BFCL ground truth value into [{func_name: {args}}] dicts.

    Handles:
      - v4 list: [{"func": {"arg": [val, ...]}}]  -- already parsed, v4 list-args
      - v3/inline string: "[{'func': {'arg': 'val'}}]"  -- Python literal string
    """
    if isinstance(gt_raw, list):
        out = []
        for item in gt_raw:
            if not isinstance(item, dict):
                continue
            normalised = {
                func_name: _normalise_gt_args(args)
                for func_name, args in item.items()
            }
            out.append(normalised)
        return out

    if isinstance(gt_raw, str):
        gt_raw = gt_raw.strip()
        if not gt_raw or gt_raw in ("[]", "None", "null"):
            return []
        try:
            parsed = ast.literal_eval(gt_raw)
            if isinstance(parsed, list):
                return _parse_ground_truth(parsed)
            if isinstance(parsed, dict):
                return _parse_ground_truth([parsed])
        except (ValueError, SyntaxError):
            pass

    return []


def _extract_messages(question_field: Any) -> List[Dict]:
    """
    Extract the messages list from a BFCL question field.

    BFCL stores questions as [[messages]] — a list of question variants;
    we take the first variant.
    """
    if isinstance(question_field, list) and question_field:
        first = question_field[0]
        if isinstance(first, list):
            return [m for m in first if isinstance(m, dict)]
        if isinstance(first, dict):
            return [first]
    if isinstance(question_field, str):
        return [{"role": "user", "content": question_field}]
    return []


def _load_category(category: str, cache_dir: Path) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Load question entries and answer lookup for one BFCL category.
    Returns (questions, {id -> ground_truth_raw}).
    """
    file_stem = CATEGORY_FILES[category]
    q_cache = cache_dir / f"{file_stem}.json"
    a_cache = cache_dir / "possible_answer" / f"{file_stem}.json"

    if not q_cache.exists():
        _download(f"{BFCL_BASE_URL}/{file_stem}.json", q_cache)

    if not a_cache.exists():
        try:
            _download(f"{BFCL_BASE_URL}/possible_answer/{file_stem}.json", a_cache)
        except RuntimeError:
            pass  # optional — entries may carry inline answers

    questions = _load_jsonl_or_array(q_cache)

    answer_lookup: Dict[str, Any] = {}
    if a_cache.exists():
        for item in _load_jsonl_or_array(a_cache):
            eid = item.get("id", "")
            gt = item.get("ground_truth", item.get("possible_answer"))
            if eid and gt is not None:
                answer_lookup[eid] = gt

    return questions, answer_lookup


def load_entries(
    categories: List[str],
    cache_dir: Path,
    samples: Optional[int] = None,
    seed: int = 42,
) -> List[BFCLEntry]:
    """Load and parse BFCL entries for the given categories."""
    for cat in categories:
        if cat not in CATEGORY_FILES:
            raise ValueError(
                f"Unknown category: {cat!r}. Valid: {sorted(CATEGORY_FILES)}"
            )

    rng = random.Random(seed)
    entries: List[BFCLEntry] = []
    skipped = 0

    for category in categories:
        questions, answer_lookup = _load_category(category, cache_dir)

        for q in questions:
            eid = q.get("id", "")
            messages = _extract_messages(q.get("question", []))
            if not messages:
                skipped += 1
                continue

            fns = [fn for fn in q.get("function", []) if isinstance(fn, dict)]
            openai_fns = [
                t
                for fn in fns
                for t in [_bfcl_fn_to_openai(fn)]
                if t is not None
            ]

            # Ground truth: check inline fields first, then separate answer file
            gt_raw = (
                q.get("ground_truth")
                or q.get("possible_answer")
                or q.get("answer")
                or answer_lookup.get(eid)
            )
            gt = _parse_ground_truth(gt_raw)
            if not gt:
                skipped += 1
                continue

            entries.append(BFCLEntry(
                id=eid,
                messages=messages,
                functions=openai_fns,
                ground_truth=gt,
                functions_bfcl=fns,
                possible_answer=gt_raw,
            ))

    if skipped:
        print(f"  Skipped {skipped} entries (empty messages or unparseable ground truth)")

    rng.shuffle(entries)
    if samples and len(entries) > samples:
        entries = entries[:samples]

    return entries


def collect_catalog(
    entries: List[BFCLEntry],
) -> Tuple[Dict[str, Dict], List[Dict[str, Any]]]:
    """
    Collect unique OpenAI-format tools across entries (name, first-seen wins).

    Returns (tools_by_name, collisions). A collision is the same function name
    with a different parameter schema; the first-seen definition is kept.
    """
    all_tools: Dict[str, Dict] = {}
    first_seen: Dict[str, str] = {}
    collisions: List[Dict[str, Any]] = []

    for entry in entries:
        for tool in entry.functions:
            name = tool["function"]["name"]
            if name not in all_tools:
                all_tools[name] = tool
                first_seen[name] = entry.id
                continue
            existing = all_tools[name]["function"].get("parameters")
            incoming = tool["function"].get("parameters")
            if json.dumps(existing, sort_keys=True, default=str) != json.dumps(
                incoming, sort_keys=True, default=str
            ):
                collisions.append({
                    "name": name,
                    "kept_entry_id": first_seen[name],
                    "dropped_entry_id": entry.id,
                    "kept_parameters": existing,
                    "dropped_parameters": incoming,
                })
    return all_tools, collisions


def collect_all_tools(entries: List[BFCLEntry]) -> Dict[str, Dict]:
    """Collect all unique tool definitions across entries (by name, first-seen wins)."""
    tools, _collisions = collect_catalog(entries)
    return tools


def write_collision_report(collisions: List[Dict[str, Any]], path: Path) -> None:
    """Write schema-collision records as JSON (no-op when empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collisions, indent=2, default=str), encoding="utf-8")


def build_shared_catalog(all_tools: Dict[str, Dict]) -> List[Dict]:
    """Return the full unique-tool catalog C used by the paper protocol."""
    return list(all_tools.values())


def build_distractor_pool(entries: List[BFCLEntry], seed: int = 42) -> List[Dict]:
    """
    Build a shuffled list of distractor tools — tools not referenced by any ground truth.

    Each entry in BFCL defines its own unique functions, so nearly every function
    ends up "required" for at least one entry. Only tools that never appear in any
    ground truth are pure distractors; the rest are distractors for OTHER entries
    (just not the one that requires them).

    Because BFCL functions are entry-local, per-instance pools (build_instance_pool)
    draw distractors from the full set excluding the entry's own functions.
    """
    required_names: set = set()
    for entry in entries:
        for call in entry.ground_truth:
            if isinstance(call, dict):
                required_names.update(call.keys())

    all_tools = collect_all_tools(entries)
    distractors = [t for n, t in all_tools.items() if n not in required_names]

    rng = random.Random(seed)
    rng.shuffle(distractors)
    return distractors


def build_instance_pool(
    entry: BFCLEntry,
    all_tools: Dict[str, Dict],
    pool_size: int,
    seed: int = 42,
) -> List[Dict]:
    """
    Build the baseline tool pool for a single evaluation instance.

    Always includes the entry's own function definitions (which contain the
    ground-truth tool). Fills remaining slots up to pool_size with tools from
    other entries, seeded per-entry for reproducibility.

    This per-instance approach avoids the global-pool overflow problem: when 598
    entries each define unique functions, a global "required" pool would have ~600
    tools (≈75 k tokens), exceeding most models' context limits. Per-instance pools
    stay at exactly pool_size tools regardless of dataset size.
    """
    entry_names = {t["function"]["name"] for t in entry.functions}
    candidate_distractors = [
        t for n, t in all_tools.items() if n not in entry_names
    ]

    n_fill = max(0, pool_size - len(entry.functions))
    rng = random.Random(seed ^ (hash(entry.id) & 0xFFFFFFFF))
    rng.shuffle(candidate_distractors)

    pool = list(entry.functions) + candidate_distractors[:n_fill]
    rng.shuffle(pool)
    return pool
