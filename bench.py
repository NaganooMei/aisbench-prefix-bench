#!/usr/bin/env python3
"""AISBench prefix workflows matching vllmbench."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import sys

from abpb import reporting, runner
from abpb.config import ConfigError, VALID_MODES, load_config


@contextmanager
def experiment_lock(work_dir):
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(work_dir) / ".abpb.lock", "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, 2)
            if not handle.tell():
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            acquire = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise ConfigError(f"Another benchmark holds {work_dir}/.abpb.lock") from exc
        try:
            yield
        finally:
            handle.seek(0)
            release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for mode in ("run", *VALID_MODES):
        command = sub.add_parser(mode)
        command.add_argument("config", nargs="?", default="bench.json")
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--results-dir", default="")
        command.add_argument("--dp-rank", type=int)
        command.add_argument(
            "--collect-metrics", action=argparse.BooleanOptionalAction, default=None
        )
        command.add_argument("--run", dest="seed_run")
    report = sub.add_parser("report")
    report.add_argument("results_dir", nargs="?", default="results")
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            reporting.write_reports(args.results_dir)
            return 0
        config = load_config(
            args.config, modes=None if args.command == "run" else [args.command]
        )
        if args.results_dir:
            config.results_dir = str(Path(args.results_dir).resolve())
        if args.dp_rank is not None:
            if any(not 0 <= args.dp_rank < service.dp for service in config.services):
                raise ConfigError("--dp-rank must be smaller than dp")
            config.services = [
                replace(service, dp_rank=args.dp_rank) for service in config.services
            ]
        if args.collect_metrics is not None:
            config.metrics_enabled = args.collect_metrics
        if args.seed_run:
            config.raw["seed_run"] = str(Path(args.seed_run).resolve())
        if args.dry_run:
            runner.run_all(config, dry_run=True)
            return 0
        with experiment_lock(config.work_dir):
            results = runner.run_all(config)
        return int(any(item.status == "error" for item in results))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
