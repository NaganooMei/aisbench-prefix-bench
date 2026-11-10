"""Terminal table and CSV summary."""

from __future__ import annotations

import csv
import json
import os

HEADERS = (
    "Service",
    "Mode",
    "Input",
    "Target",
    "HBM hit",
    "Ext hit",
    "Hit tok",
    "Prefill tok",
    "TTFT ms",
    "TPOT ms",
    "Out tok/s",
    "Total tok/s",
    "Load GB/s",
    "Status",
)


def _hit_tokens(hits: dict) -> float | None:
    hbm = hits.get("hbm_hits")
    external = hits.get("external_hits")
    if hbm is None and external is None:
        return None
    return (hbm or 0.0) + (external or 0.0)


def _prefill_tokens(perf: dict, hits: dict) -> float | None:
    cached = _hit_tokens(hits)
    total = perf.get("input_tokens")
    if cached is None or total is None:
        return None
    return max(total - cached, 0.0)


def _rate(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def _input_tokens(case_label: str) -> str:
    # labels look like in8192_out1_hit0.90_c8_n16
    for part in case_label.split("_"):
        if part.startswith("in") and part[2:].isdigit():
            return f"{int(part[2:]) // 1024}K"
    return case_label


def row_for(result) -> list[str]:
    perf = result.perf or {}
    hits = result.hits or {}
    status = result.status
    if status == "error":
        status = "error: " + (result.error or "")[:60]
    return [
        result.service,
        result.mode,
        _input_tokens(result.case),
        f"{result.target_hit * 100:.0f}%",
        _rate(hits.get("hbm_hit_rate")),
        _rate(hits.get("external_hit_rate")),
        _num(_hit_tokens(hits), 0),
        _num(_prefill_tokens(perf, hits), 0),
        _num(perf.get("ttft")),
        _num(perf.get("tpot")),
        _num(perf.get("output_throughput")),
        _num(perf.get("total_throughput")),
        _num(hits.get("ucm_load_gbps"), 2),
        status,
    ]


def render(results) -> str:
    rows = [row_for(r) for r in results]
    widths = [len(h) for h in HEADERS]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(HEADERS))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def write_csv(path: str, results) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADERS)
        for result in results:
            writer.writerow(row_for(result))
    return path


class _Loaded:
    """Minimal shim so report can render results read back from disk."""

    def __init__(self, payload: dict):
        self.service = payload.get("service", "")
        self.case = payload.get("case", "")
        self.mode = payload.get("mode", "")
        self.target_hit = payload.get("target_hit", 0.0) or 0.0
        self.perf = payload.get("perf") or {}
        self.hits = payload.get("hits") or {}
        self.status = payload.get("status", "ok")
        self.error = payload.get("error", "")


def load_results(results_dir: str) -> list[_Loaded]:
    found = []
    for root, _dirs, files in os.walk(results_dir):
        for name in files:
            if name != "case_result.json":
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as fh:
                found.append((os.path.getmtime(path), _Loaded(json.load(fh))))
    found.sort(key=lambda item: item[0])
    return [item[1] for item in found]
