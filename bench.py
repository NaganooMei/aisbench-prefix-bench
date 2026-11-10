#!/usr/bin/env python3
"""aisbench-prefix-bench: prefix-cache hit testing on top of AISBench."""

from __future__ import annotations

import argparse
import os
import sys

from abpb import reporting, runner
from abpb.config import ConfigError, load_config


def cmd_run(args) -> int:
    try:
        config = load_config(args.config)
    except (OSError, ConfigError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.results_dir:
        config.results_dir = os.path.abspath(args.results_dir)

    results = runner.run_all(config, log=print, dry_run=args.dry_run)
    if args.dry_run or not results:
        return 0

    table = reporting.render(results)
    print()
    print(table)

    os.makedirs(config.results_dir, exist_ok=True)
    csv_path = reporting.write_csv(os.path.join(config.results_dir, "summary.csv"), results)
    print(f"\nsummary: {csv_path}")
    if config.raw.get("aisbench"):
        print(f"run dir: {config.work_dir}")
    errors = [r for r in results if r.status == "error"]
    if errors:
        print(f"\n{len(errors)} case(s) failed:", file=sys.stderr)
        for item in errors:
            print(f"  {item.service} {item.case} [{item.mode}]: {item.error}", file=sys.stderr)
        return 1
    return 0


def cmd_report(args) -> int:
    results = reporting.load_results(args.results_dir)
    if not results:
        print(f"no case_result.json under {args.results_dir}", file=sys.stderr)
        return 2
    print(reporting.render(results))
    out = reporting.write_csv(os.path.join(args.results_dir, "summary.csv"), results)
    print(f"\nsummary: {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run the configured matrix")
    run_parser.add_argument("config", help="path to bench.json")
    run_parser.add_argument("--dry-run", action="store_true", help="validate and print the plan")
    run_parser.add_argument("--results-dir", default="", help="override results directory")
    run_parser.set_defaults(func=cmd_run)

    report_parser = sub.add_parser("report", help="re-render an existing results tree")
    report_parser.add_argument("results_dir")
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
