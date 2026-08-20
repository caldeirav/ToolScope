#!/usr/bin/env python3
"""
BFCL evaluation: LLM tool-calling accuracy across a model × retriever matrix.

Usage (from the project root):
  python eval/run_eval.py                          # full matrix, all configured models
  python eval/run_eval.py --model Qwen/Qwen2.5-7B-Instruct  # single model override
  python eval/run_eval.py --dry-run --samples 20   # no GPU required

Run `python eval/run_eval.py --help` for all options.
"""

# ── Path bootstrap (must precede all local imports) ───────────────────────────
import os
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent     # .../eval/
_SRC_DIR = _EVAL_DIR.parent / "src"             # .../src/

for _p in [str(_EVAL_DIR), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# On Apple Silicon, remove PyTorch MPS's self-imposed memory cap so it can use
# whatever physical unified memory is actually available.  Without this, MPS
# refuses new allocations even when physical memory exists but "other" (CPU-side)
# allocations have grown to crowd the default watermark.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import gc
import json
import time
import traceback
from datetime import datetime
import yaml
from tqdm import tqdm

import toolscope
from bfcl_eval.checkpoint import CheckpointManager
from bfcl_eval.dataset import load_entries, collect_all_tools, build_instance_pool
from bfcl_eval.model import DummyModel, HFModel
from bfcl_eval.evaluate import evaluate_instance, aggregate, AggregateMetrics, RetrieverMetrics
from bfcl_eval.report import (
    print_report, print_cross_model_summary, save_results, print_verbose_instance,
)
from bfcl_eval.retrieval import RandomRetriever, BM25Retriever, TFIDFRetriever, OracleRetriever


# ── Config helpers ──────────────────────────────────────────────────────────


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_set(d: dict, dotted_key: str, value: object) -> None:
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _resolve_model_names(model_cfg: dict, cli_model: str | None) -> list[str]:
    """Return the ordered list of model names to evaluate.

    Priority: --model CLI flag > model.names list > model.name (legacy).
    """
    if cli_model:
        return [cli_model]
    names = model_cfg.get("names")
    if names:
        return list(names)
    legacy = model_cfg.get("name")
    if legacy:
        return [legacy]
    return ["Qwen/Qwen2.5-7B-Instruct"]


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.device:
        _deep_set(cfg, "model.device", args.device)
    if args.dtype:
        _deep_set(cfg, "model.dtype", args.dtype)
    if args.samples is not None:
        _deep_set(cfg, "dataset.samples", args.samples)
    if args.pool_size is not None:
        _deep_set(cfg, "dataset.pool_size", args.pool_size)
    if args.k is not None:
        _deep_set(cfg, "toolscope.k", args.k)
    if args.category:
        _deep_set(cfg, "dataset.categories", args.category)
    if args.seed is not None:
        _deep_set(cfg, "dataset.seed", args.seed)
    if args.output_dir:
        _deep_set(cfg, "output.results_dir", args.output_dir)
    if args.verbose:
        _deep_set(cfg, "output.verbose", True)
    return cfg


_GATED_MODELS = {
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def _preflight_check(model_names: list[str]) -> None:
    """Warn early about prerequisites that would cause the run to fail later."""
    import os

    gated = [m for m in model_names if m in _GATED_MODELS]
    if not gated:
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return

    print("━" * 60)
    print("  PRE-FLIGHT WARNING")
    print("━" * 60)
    print("  The following model(s) are gated on HuggingFace and")
    print("  require authentication:")
    for m in gated:
        print(f"    • {m}")
    print()
    print("  To fix this before starting the run:")
    print("    1. Accept the model licence at huggingface.co/<model>")
    print("    2. Set your HF token:  export HF_TOKEN=hf_...")
    print()
    print("  Without a token, these models will fail at load time.")
    print("  The run will continue with the remaining models, but")
    print("  you will need to re-run those models separately.")
    print("━" * 60)
    print()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BFCL evaluation: LLM tool calling across a model × retriever matrix"
    )
    p.add_argument("--config", default="eval/config.yaml")
    p.add_argument("--model",
                   help="Single HuggingFace model name (overrides model.names in config)")
    p.add_argument("--device", help="Inference device: auto | cpu | cuda | mps")
    p.add_argument("--dtype", help="Model dtype: auto | float16 | bfloat16 | float32")
    p.add_argument("--samples", type=int, help="Max evaluation instances (per category)")
    p.add_argument("--pool-size", dest="pool_size", type=int)
    p.add_argument("--k", type=int, help="Tools each retriever returns per query")
    p.add_argument("--category", nargs="+",
                   help="BFCL categories: simple multiple parallel parallel_multiple")
    p.add_argument("--seed", type=int)
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--dry-run", action="store_true",
                   help="Use a dummy model (no GPU needed) to test the full pipeline")
    p.add_argument("--no-resume", dest="no_resume", action="store_true",
                   help="Ignore any existing checkpoint and start from scratch")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-instance details during evaluation")
    return p


def _write_error_log(
    output_dir: Path,
    model_name: str,
    instance_id: str,
    exc_type: str,
    exc_msg: str,
    tb: str,
) -> None:
    """Append a structured error record to a per-model error log file."""
    try:
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        slug = model_name.split("/")[-1]
        log_path = log_dir / f"{slug}_errors.log"
        with open(log_path, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{ts}] {instance_id}  {exc_type}: {exc_msg}\n")
            f.write(tb)
            f.write("\n" + "-" * 60 + "\n")
    except Exception:
        pass  # log failure must never interrupt the eval loop


def _flush_accelerator_memory() -> None:
    """Return cached GPU/MPS buffers to the system allocator.

    On MPS (Apple Silicon), PyTorch keeps Metal buffers in a free-list after
    tensors go out of scope.  Without explicit flushing the free-list grows
    across generate() calls, fragmenting memory and making each new allocation
    progressively slower.  Call this once per instance (not per predict call)
    to balance cleanup overhead vs. allocation pressure.
    """
    gc.collect()
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _free_model(model: object) -> None:
    """Delete a model and try to release accelerator memory."""
    del model
    _flush_accelerator_memory()


def _short_name(model_name: str) -> str:
    """Last path component, e.g. 'Qwen/Qwen2.5-7B-Instruct' → 'Qwen2.5-7B-Instruct'."""
    return model_name.split("/")[-1]


def _load_metrics_from_json(path: Path) -> AggregateMetrics:
    """Reconstruct AggregateMetrics from a saved result JSON written by save_results()."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    m = data["metrics"]
    return AggregateMetrics(
        n=m["n"],
        n_skipped=m["n_skipped"],
        baseline_name_acc=m["baseline_name_acc"],
        baseline_exact_match=m["baseline_exact_match"],
        mean_baseline_tokens=m["mean_baseline_tokens"],
        retrievers={
            name: RetrieverMetrics(**rm)
            for name, rm in m["retrievers"].items()
        },
    )


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _build_parser().parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = _EVAL_DIR / "config.yaml"
    if not cfg_path.exists():
        print(f"Config not found at {args.config}. Run from the project root.")
        sys.exit(1)

    cfg = _load_config(cfg_path)
    cfg = _apply_overrides(cfg, args)

    model_cfg   = cfg.get("model", {})
    ds_cfg      = cfg.get("dataset", {})
    ts_cfg      = cfg.get("toolscope", {})
    out_cfg     = cfg.get("output", {})
    embed_cfg   = ts_cfg.get("embedding", {})

    model_names       = _resolve_model_names(model_cfg, args.model)
    categories        = ds_cfg.get("categories", ["multiple"])
    samples           = ds_cfg.get("samples")
    pool_size         = ds_cfg.get("pool_size", 100)
    seed              = ds_cfg.get("seed", 42)
    k                 = ts_cfg.get("k", 5)
    embed_model       = embed_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    embed_provider    = embed_cfg.get("provider", "sentence-transformers")
    embed_allow_dl    = embed_cfg.get("allow_download", True)
    output_dir        = Path(out_cfg.get("results_dir", "eval/results"))
    verbose           = out_cfg.get("verbose", False)
    cache_dir         = Path(ds_cfg.get("cache_dir", "eval/.bfcl_cache"))

    print()
    print("=== BFCL Evaluation — Model × Retriever Matrix ===")
    if args.dry_run:
        print("  Mode      : DRY RUN (dummy model, no GPU needed)")
    print(f"  Models    : {len(model_names)}")
    for mn in model_names:
        print(f"              {mn}")
    print(f"  Categories: {categories}")
    print(f"  Samples   : {samples or 'all'}")
    print(f"  Pool size : {pool_size}  |  k: {k}  |  Seed: {seed}")
    print(f"  Embedder  : {embed_model}")
    print()

    # ══ Shared setup ═════════════════════════════════════════════════════════

    print("Loading BFCL data...")
    entries = load_entries(
        categories=categories, cache_dir=cache_dir, samples=samples, seed=seed,
    )
    print(f"  {len(entries)} instances loaded")
    if not entries:
        print("No entries loaded. Check category names or network access.")
        sys.exit(1)

    print("Collecting tool definitions...")
    all_tools = collect_all_tools(entries)
    print(f"  {len(all_tools)} unique tools across all entries")
    print()

    print("Building ToolScope index...")
    embedding_config = toolscope.EmbeddingConfig(
        provider=embed_provider,
        model=embed_model,
        allow_download=embed_allow_dl,
        normalize=True,
    )
    ts_index = toolscope.index(list(all_tools.values()), embedding=embedding_config)
    print("  Index ready.")
    print()

    print("Building retrieval baselines...")
    tool_list = list(all_tools.values())
    retrievers = {
        "Random":    RandomRetriever(tool_list, seed=seed),
        "BM25":      BM25Retriever(tool_list),
        "TF-IDF":    TFIDFRetriever(tool_list),
        "Oracle*":   OracleRetriever(tool_list, seed=seed),
        "ToolScope": ts_index,
    }
    print(f"  Retrievers: {', '.join(retrievers)}")
    print()

    # ══ Pre-flight checks ════════════════════════════════════════════════════

    _preflight_check(model_names)

    # ══ Model loop ═══════════════════════════════════════════════════════════

    all_metrics: dict[str, AggregateMetrics] = {}   # model_name → metrics
    failed_models: list[str] = []

    for model_idx, model_name in enumerate(model_names):

        t0 = time.monotonic()
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("─" * 60)
        print(f"  Model {model_idx + 1}/{len(model_names)}: {model_name}")
        print(f"  Started   : {started_at}")
        print("─" * 60)
        print()

        try:
            with CheckpointManager(
                output_dir=output_dir,
                model_name=model_name,
                categories=categories,
                pool_size=pool_size,
                seed=seed,
                k=k,
                resume=not args.no_resume,
                dry_run=args.dry_run,
            ) as ckpt:

                # ── Resume: load any previously evaluated instances ──────────
                done = ckpt.load()
                done_ids = set(done.keys())
                remaining = [e for e in entries if e.id not in done_ids]

                # In an exec chain, models completed in a prior process invocation
                # have nothing left to do.  Skip silently — metrics are reconstructed
                # from the saved JSON file at cross-model summary time.
                if not remaining and not args.model and len(model_names) > 1:
                    continue

                print(f"  Checkpoint: {ckpt.path}")
                if done:
                    print(
                        f"  Resuming  : {len(done)} done, "
                        f"{len(remaining)} remaining"
                    )
                else:
                    print(f"  Resuming  : fresh start ({len(entries)} instances)")
                print()

                # ── Load model only if there is work left to do ──────────────
                model = None
                if remaining:
                    if args.dry_run:
                        model = DummyModel()
                    else:
                        model = HFModel(
                            model_name=model_name,
                            device=model_cfg.get("device", "auto"),
                            dtype=model_cfg.get("dtype", "auto"),
                            max_new_tokens=model_cfg.get("max_new_tokens", 256),
                        )
                    # Eager load: any failure here is caught at model scope,
                    # not silently swallowed instance-by-instance.
                    model.load()
                else:
                    print("  All instances already evaluated — skipping model load.")

                results = list(done.values())
                n_skipped = 0

                if remaining:
                    for entry in tqdm(
                        remaining,
                        desc=_short_name(model_name),
                        unit="inst",
                        initial=len(done),
                        total=len(entries),
                    ):
                        instance_pool = build_instance_pool(
                            entry=entry, all_tools=all_tools, pool_size=pool_size, seed=seed,
                        )
                        result = None
                        try:
                            result = evaluate_instance(
                                entry_id=entry.id,
                                messages=entry.messages,
                                ground_truth=entry.ground_truth,
                                tool_pool=instance_pool,
                                model=model,
                                retrievers=retrievers,
                                k=k,
                            )
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            tb = traceback.format_exc()
                            tqdm.write(
                                f"\n  WARNING: {entry.id} failed — skipping\n"
                                f"  {type(exc).__name__}: {exc}\n"
                                f"{tb}"
                            )
                            _write_error_log(
                                output_dir, model_name, entry.id,
                                type(exc).__name__, str(exc), tb,
                            )

                        if result is None:
                            n_skipped += 1
                        else:
                            results.append(result)
                            ckpt.write(result)
                            if verbose:
                                print_verbose_instance(result, k)

                        # Flush GPU/MPS allocator free-list after every instance
                        # (success or failure).  On Apple Silicon, PyTorch keeps
                        # Metal buffers in a free-list after tensors go out of
                        # scope; without periodic flushing the free-list grows
                        # across generate() calls and inference slows to a crawl.
                        _flush_accelerator_memory()

                    if model is not None:
                        _free_model(model)

        except KeyboardInterrupt:
            print(f"\n  Interrupted during {model_name}. Checkpoint is intact.")
            raise
        except Exception as exc:
            failed_models.append(model_name)
            print(f"\n  Model {model_name} failed: {type(exc).__name__}: {exc}")
            print("  Continuing to the next model.\n")
            continue

        elapsed_s = time.monotonic() - t0
        h, rem = divmod(int(elapsed_s), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
        finished_at = datetime.now().strftime("%H:%M:%S")
        print(f"  Completed in {elapsed_str}  (finished {finished_at})")
        print()

        metrics = aggregate(results, n_skipped=n_skipped)
        all_metrics[model_name] = metrics

        print_report(
            metrics=metrics,
            model_name=model_name,
            categories=categories,
            pool_size=pool_size,
            k=k,
            embedding_model=embed_model,
        )

        save_results(
            results=results,
            metrics=metrics,
            config=cfg,
            output_dir=output_dir,
            model_name=model_name,
        )
        print()

        # After each model (except the last), replace this process image with a
        # fresh Python interpreter for the next model.  os.execv() reclaims ALL
        # memory — Python heap, MPS Metal buffers, macOS compressor pages — so
        # the next model always starts with the full memory budget.  Unlike a
        # subprocess, there is zero coordinator overhead: there is no second
        # process sitting in memory alongside the worker.
        # Checkpoints ensure no work is lost across the exec chain.
        # --no-resume is stripped so exec'd invocations don't delete checkpoints
        # that were built by earlier links in the chain.
        if not args.model and model_idx < len(model_names) - 1:
            exec_argv = [str(Path(__file__).resolve())] + [
                a for a in sys.argv[1:] if a != "--no-resume"
            ]
            os.execv(sys.executable, [sys.executable] + exec_argv)
            # never reached

    # ══ Cross-model summary (only meaningful with ≥2 models) ════════════════

    if failed_models:
        print(f"  Note: {len(failed_models)} model(s) failed and were skipped:")
        for fm in failed_models:
            print(f"    • {fm}")
        print()

    # Load metrics from models that ran in a prior process invocation (exec
    # chain).  The last process in the chain only has its own model in
    # all_metrics; earlier models' metrics live in their saved JSON files.
    for mname in model_names:
        if mname not in all_metrics:
            slug = mname.split("/")[-1].replace(" ", "_")
            candidates = sorted(
                output_dir.glob(f"bfcl_eval_{slug}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                try:
                    all_metrics[mname] = _load_metrics_from_json(candidates[0])
                except Exception as exc:
                    print(f"  Warning: could not load metrics for {mname}: {exc}")

    if len(all_metrics) >= 2:
        print_cross_model_summary(
            all_metrics=all_metrics,
            k=k,
        )


if __name__ == "__main__":
    main()
