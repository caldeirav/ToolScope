"""
Incremental checkpoint system for evaluation runs.

Writes each InstanceResult to a JSONL file immediately after it is computed
and flushed to disk. On restart, loads previously saved results so the
evaluation loop can skip already-completed instances.

Checkpoint path: {output_dir}/checkpoints/{model_slug}_{config_sig}.jsonl

The config signature is a short hash of (model_name, categories, pool_size,
seed, k).  If any of these change, the signature changes and the old
checkpoint is ignored rather than silently loaded.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, IO, List, Optional

from .evaluate import InstanceResult, RetrieverResult
from .model import ParsedToolCall


# ── Config signature ─────────────────────────────────────────────────────────


def _config_sig(
    model_name: str,
    categories: List[str],
    pool_size: int,
    seed: int,
    k: int,
    dry_run: bool = False,
    protocol: str = "legacy",
    catalog_size: int = 0,
) -> str:
    prefix = "dryrun|" if dry_run else ""
    key = (
        f"{prefix}{protocol}|{model_name}|{'|'.join(sorted(categories))}"
        f"|{pool_size}|{catalog_size}|{seed}|{k}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ── Deserialization ──────────────────────────────────────────────────────────


def _parsed_tool_call_from_dict(d: Optional[Dict]) -> Optional[ParsedToolCall]:
    if d is None:
        return None
    return ParsedToolCall(name=d["name"], arguments=d.get("arguments", {}))


def _retriever_result_from_dict(d: Dict) -> RetrieverResult:
    return RetrieverResult(
        name_acc=d["name_acc"],
        exact_match=d["exact_match"],
        ast_acc=d.get("ast_acc", d.get("exact_match", False)),
        recall=d["recall"],
        dcg=d["dcg"],
        ndcg=d["ndcg"],
        gt_rank=d.get("gt_rank"),
        tool_names=d["tool_names"],
        tokens=d["tokens"],
        compression_rate=d["compression_rate"],
        raw=d.get("raw", ""),
        predicted=_parsed_tool_call_from_dict(d.get("predicted")),
        error=d.get("error"),
        latency_ms=d.get("latency_ms", 0.0),
        prompt_tokens=d.get("prompt_tokens"),
    )


def _instance_result_from_dict(d: Dict) -> InstanceResult:
    return InstanceResult(
        id=d["id"],
        query=d["query"],
        ground_truth_names=d["ground_truth_names"],
        baseline_name_acc=d["baseline_name_acc"],
        baseline_exact_match=d["baseline_exact_match"],
        baseline_ast_acc=d.get("baseline_ast_acc", d.get("baseline_exact_match", False)),
        baseline_tokens=d["baseline_tokens"],
        baseline_raw=d.get("baseline_raw", ""),
        baseline_pred=_parsed_tool_call_from_dict(d.get("baseline_pred")),
        baseline_error=d.get("baseline_error"),
        baseline_latency_ms=d.get("baseline_latency_ms", 0.0),
        baseline_prompt_tokens=d.get("baseline_prompt_tokens"),
        retrievers={
            name: _retriever_result_from_dict(rr)
            for name, rr in d.get("retrievers", {}).items()
        },
    )


# ── CheckpointManager ────────────────────────────────────────────────────────


class CheckpointManager:
    """
    Per-model incremental JSONL checkpoint.

    Usage (context manager):

        with CheckpointManager(output_dir, model_name, ...) as ckpt:
            done     = ckpt.load()          # Dict[id, InstanceResult]
            done_ids = set(done.keys())
            results  = list(done.values())

            for entry in entries:
                if entry.id in done_ids:
                    continue                 # skip already-evaluated instances
                result = evaluate_instance(...)
                if result:
                    results.append(result)
                    ckpt.write(result)       # flushed to disk immediately
    """

    def __init__(
        self,
        output_dir: Path,
        model_name: str,
        categories: List[str],
        pool_size: int,
        seed: int,
        k: int,
        resume: bool = True,
        dry_run: bool = False,
        protocol: str = "legacy",
        catalog_size: int = 0,
    ) -> None:
        slug = model_name.split("/")[-1]
        sig = _config_sig(
            model_name, categories, pool_size, seed, k,
            dry_run=dry_run, protocol=protocol, catalog_size=catalog_size,
        )
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._path = ckpt_dir / f"{slug}_{sig}.jsonl"
        self._resume = resume
        self._file: Optional[IO] = None

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Dict[str, InstanceResult]:
        """
        Load previously saved results.

        Returns an empty dict when resume=False or the checkpoint file does
        not exist.  Silently skips malformed lines (e.g. truncated by a crash
        mid-write) so a partial last line never blocks recovery.

        When resume=False, deletes any existing checkpoint file immediately
        so a future run without --no-resume won't accidentally inherit stale data.
        """
        results: Dict[str, InstanceResult] = {}
        if not self._resume:
            if self._path.exists():
                self._path.unlink()
            return results
        if not self._path.exists():
            return results
        with open(self._path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    r = _instance_result_from_dict(d)
                    results[r.id] = r
                except Exception as exc:
                    print(
                        f"  Warning: skipping malformed checkpoint line {line_no}: {exc}"
                    )
        return results

    def write(self, result: InstanceResult) -> None:
        """Append one result and flush to disk. Opens the file on first call."""
        if self._file is None:
            # Append when resuming (preserve prior results); overwrite on fresh start.
            mode = "a" if self._resume else "w"
            self._file = open(self._path, mode, encoding="utf-8")
        self._file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()
