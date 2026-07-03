"""The ``mizan`` command-line interface.

``mizan run`` loads and validates a YAML config, creates a timestamped run
folder with reproducibility metadata (resolved config, model ids, UTC timestamp,
tool version), then executes the evaluation loop and writes ``results.json``.
``--dry-run`` stops after creating the folder, performing no model calls.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from mizan import __version__
from mizan.config import ConfigError, RunConfig, load_config
from mizan.dataset import DatasetError, load_items
from mizan.results import RunMetadata
from mizan.runner import run_eval

DEFAULT_CACHE_DIR = ".mizan_cache"


def _mizan_version() -> str:
    try:
        return version("mizan-evals")
    except PackageNotFoundError:
        return __version__


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "run"


def _make_run_metadata(config: RunConfig, now: datetime) -> RunMetadata:
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{_slugify(config.name)}"
    return RunMetadata(
        run_id=run_id,
        name=config.name,
        created_at=now.isoformat(),
        mizan_version=_mizan_version(),
        model_ids=config.model_ids,
        config=config,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Handle ``mizan run --config <path>``."""
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir)
    metadata = _make_run_metadata(config, datetime.now(UTC))
    run_dir = output_dir / metadata.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    print(f"run id     : {metadata.run_id}")
    print(f"created_at : {metadata.created_at}")
    print(f"version    : {metadata.mizan_version}")
    print(f"run dir    : {run_dir}")
    print(f"models     : {', '.join(config.model_ids)}")
    print(f"tasks      : {', '.join(t.value for t in config.tasks)}")
    print(f"languages  : {', '.join(lang.value for lang in config.languages)}")

    if args.dry_run:
        print("note       : dry run - configuration validated, no model calls made.")
        return 0

    try:
        items = load_items(config.dataset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = run_eval(config, items, metadata, cache_root=Path(args.cache_dir))
    (run_dir / "results.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    summary = result.summary
    print(
        f"results    : {summary['n_results']} "
        f"({summary['n_cached']} cached, {summary['n_errors']} errors)"
    )
    if summary["retrieval_pending"]:
        print(
            f"pending    : {summary['retrieval_pending']} retrieval item(s) "
            "await the retrieval eval (later phase)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``mizan`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mizan",
        description="Evaluation harness for Arabic RAG and agent tool-calling.",
    )
    parser.add_argument("--version", action="version", version=f"mizan {_mizan_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an evaluation from a config")
    run_parser.add_argument("--config", required=True, help="path to a YAML run configuration")
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="where to create the run folder (overrides output_dir in the config)",
    )
    run_parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"directory for the resumable LLM cache (default: {DEFAULT_CACHE_DIR})",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config and create the run folder without calling any model",
    )
    run_parser.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``mizan`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
