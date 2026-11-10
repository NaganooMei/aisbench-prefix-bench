"""Configuration parsing and validation for aisbench-prefix-bench."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


class ConfigError(ValueError):
    """Raised when bench.json cannot be used as written."""


@dataclass(frozen=True)
class Service:
    name: str
    base_url: str
    served_model: str
    model_path: str
    dp: int = 1
    api_key: str = ""

    @property
    def tokenizer_path(self) -> str:
        return self.model_path


@dataclass(frozen=True)
class Case:
    input_len: int
    output_len: int
    hit_rate: float
    concurrency: int
    num_requests: int
    prefix_num: int | None = None
    seed: int | None = None
    request_rate: float = 0.0

    def prefix_groups(self) -> int:
        """Distinct prefixes: one per request unless a shared count is pinned."""
        if self.prefix_num is None:
            return self.num_requests
        return min(self.prefix_num, self.num_requests)

    def prefix_tokens(self, block_tokens: int) -> int:
        """Prefix length rounded down to a multiple of the cache block size."""
        raw = int(self.input_len * self.hit_rate)
        return (raw // block_tokens) * block_tokens

    def suffix_tokens(self, block_tokens: int) -> int:
        return self.input_len - self.prefix_tokens(block_tokens) - UNIQUE_TOKENS

    def label(self, block_tokens: int) -> str:
        return (
            f"in{self.input_len}_out{self.output_len}"
            f"_hit{self.prefix_tokens(block_tokens) / self.input_len:.2f}"
            f"_c{self.concurrency}_n{self.num_requests}"
        )


UNIQUE_TOKENS = 3


@dataclass
class Config:
    services: list[Service]
    cases: list[Case]
    modes: list[str]
    block_tokens: int
    work_dir: str
    results_dir: str
    configs_dir: str
    summarizer: str
    aisbench_command: str
    aisbench_extra_args: list[str]
    corpus_path: str
    seed: int | None
    metrics_enabled: bool
    metrics_settle: float
    metrics_timeout: float
    reset_timeout: float
    reset_interval: float
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


VALID_MODES = ("hbm", "cache", "posix-reset")


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise ConfigError(f"{where}: missing required field '{key}'")
    return d[key]


def _positive_int(d: dict, key: str, where: str, default: int | None = None) -> int:
    value = d.get(key, default)
    if value is None:
        raise ConfigError(f"{where}: missing required field '{key}'")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{where}: '{key}' must be a positive integer, got {value!r}")
    return value


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a JSON object")

    base = os.path.dirname(os.path.abspath(path))

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base, p)

    services_raw = raw.get("services")
    if not isinstance(services_raw, list) or not services_raw:
        raise ConfigError("services: must be a non-empty list")
    services = []
    for i, item in enumerate(services_raw):
        where = f"services[{i}]"
        name = _require(item, "name", where)
        services.append(
            Service(
                name=str(name),
                base_url=str(_require(item, "base_url", where)).rstrip("/"),
                served_model=str(_require(item, "served_model", where)),
                model_path=resolve(str(_require(item, "model_path", where))),
                dp=_positive_int(item, "dp", where, 1),
                api_key=str(item.get("api_key", "")),
            )
        )
    names = [s.name for s in services]
    if len(set(names)) != len(names):
        raise ConfigError(f"services: duplicate names in {names}")

    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ConfigError("cases: must be a non-empty list")
    cases = []
    for i, item in enumerate(cases_raw):
        where = f"cases[{i}]"
        hit = float(_require(item, "hit_rate", where))
        if not (0 < hit < 1):
            raise ConfigError(f"{where}: hit_rate must be between 0 and 1 (exclusive), got {hit}")
        case = Case(
            input_len=_positive_int(item, "input_len", where),
            output_len=_positive_int(item, "output_len", where),
            hit_rate=hit,
            concurrency=_positive_int(item, "concurrency", where),
            num_requests=_positive_int(item, "num_requests", where),
            prefix_num=(
                _positive_int(item, "prefix_num", where) if "prefix_num" in item else None
            ),
            seed=item.get("seed"),
            request_rate=float(item.get("request_rate", 0) or 0),
        )
        if case.concurrency > case.num_requests:
            raise ConfigError(f"{where}: concurrency exceeds num_requests")
        cases.append(case)

    block_tokens = _positive_int(raw, "block_tokens", "root", 1)
    for i, case in enumerate(cases):
        if case.prefix_tokens(block_tokens) < block_tokens:
            raise ConfigError(
                f"cases[{i}]: hit_rate too small for block_tokens={block_tokens} "
                f"(prefix would be {case.prefix_tokens(block_tokens)} tokens)"
            )
        if case.suffix_tokens(block_tokens) < 1:
            raise ConfigError(
                f"cases[{i}]: hit_rate too high for input_len={case.input_len}"
            )

    modes = raw.get("modes", list(VALID_MODES))
    unknown = [m for m in modes if m not in VALID_MODES]
    if unknown:
        raise ConfigError(f"modes: unknown {unknown}, expected a subset of {VALID_MODES}")

    metrics = raw.get("metrics", {})
    reset = raw.get("reset", {})
    aisbench = raw.get("aisbench", {})

    config = Config(
        services=services,
        cases=cases,
        modes=list(modes),
        block_tokens=block_tokens,
        work_dir=resolve(str(raw.get("work_dir", "work"))),
        results_dir=resolve(str(raw.get("results_dir", "results"))),
        configs_dir=resolve(str(aisbench.get("configs_dir", ""))) if aisbench.get("configs_dir") else "",
        summarizer=str(aisbench.get("summarizer", "default_perf")),
        aisbench_command=str(aisbench.get("command", "ais_bench")),
        aisbench_extra_args=[str(a) for a in aisbench.get("extra_args", [])],
        corpus_path=resolve(str(raw["corpus_path"])) if raw.get("corpus_path") else "",
        seed=raw.get("seed"),
        metrics_enabled=bool(metrics.get("enabled", True)),
        metrics_settle=float(metrics.get("settle_seconds", 2)),
        metrics_timeout=float(metrics.get("timeout_seconds", 10)),
        reset_timeout=float(reset.get("timeout_seconds", 60)),
        reset_interval=float(reset.get("retry_interval_seconds", 5)),
        source_path=os.path.abspath(path),
        raw=raw,
    )
    return config
