"""Configuration parsing and validation for aisbench-prefix-bench."""

from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass, field
from decimal import Decimal
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
    dp_rank: int | None = None

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

    def prefix_tokens(self, block_tokens: int, mode: str = "hbm") -> int:
        """Prefix length rounded down to a multiple of the cache block size."""
        if mode == "hbm-no-cache":
            return 0
        tail = 32 if mode in ("v4hbm", "hbm-shared-v4") else 1
        raw = min(
            int(self.input_len * Decimal(str(self.hit_rate))),
            max(0, self.input_len - tail),
        )
        alignment = 4096 if tail == 32 else block_tokens
        target = raw // alignment * alignment
        return target + 32 if target and tail == 32 else target

    def suffix_tokens(self, block_tokens: int) -> int:
        return self.input_len - self.prefix_tokens(block_tokens)

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


VALID_MODES = (
    "hbm",
    "hbm-shared",
    "hbm-shared-v4",
    "v4hbm",
    "hbm-no-cache",
    "cache",
    "posix-seed",
    "posix-run",
    "posix-reset",
)


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


def load_config(path: str, modes: list[str] | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a JSON object")
    if modes is not None:
        raw["modes"] = modes

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
                dp_rank=item.get(
                    "dp_rank",
                    (
                        raw.get("routing", {}).get("rank")
                        if raw.get("routing", {}).get("mode") == "fixed"
                        else None
                    ),
                ),
            )
        )
    names = [s.name for s in services]
    if len(set(names)) != len(names):
        raise ConfigError(f"services: duplicate names in {names}")
    for service in services:
        if service.dp_rank is not None and (
            isinstance(service.dp_rank, bool)
            or not isinstance(service.dp_rank, int)
            or not 0 <= service.dp_rank < service.dp
        ):
            raise ConfigError("dp_rank must be an integer in [0, dp)")

    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ConfigError("cases: must be a non-empty list")
    cases = []
    for i, item in enumerate(cases_raw):
        where = f"cases[{i}]"
        hit = float(
            item.get("hit_rate", 0)
            if raw.get("modes") == ["hbm-no-cache"]
            else _require(item, "hit_rate", where)
        )
        if not (0 <= hit <= 1):
            raise ConfigError(f"{where}: hit_rate must be between 0 and 1, got {hit}")
        case = Case(
            input_len=_positive_int(item, "input_len", where),
            output_len=_positive_int(item, "output_len", where),
            hit_rate=hit,
            concurrency=_positive_int(item, "concurrency", where),
            num_requests=_positive_int(item, "num_requests", where),
            prefix_num=(
                _positive_int(item, "prefix_num", where)
                if "prefix_num" in item
                else None
            ),
            seed=item.get("seed"),
            request_rate=float(item.get("request_rate", 0) or 0),
        )
        if case.concurrency > case.num_requests:
            raise ConfigError(f"{where}: concurrency exceeds num_requests")
        cases.append(case)

    block_tokens = _positive_int(raw, "block_tokens", "root", 1)
    for seed in [raw.get("seed"), *(case.seed for case in cases)]:
        if (
            seed is not None
            and seed != "random"
            and (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed < 2**32
            )
        ):
            raise ConfigError("seed must be random or an integer in [0, 2**32)")
    modes = raw.get("modes", ["hbm", "cache", "posix-reset"])
    unknown = [m for m in modes if m not in VALID_MODES]
    if unknown:
        raise ConfigError(
            f"modes: unknown {unknown}, expected a subset of {VALID_MODES}"
        )
    if "posix-seed" in modes and "posix-run" in modes:
        raise ConfigError(
            "Run posix-seed and posix-run separately, with a service restart between them"
        )

    metrics = raw.get("metrics", {})
    if metrics.get("collector_path"):
        metrics["collector_path"] = resolve(metrics["collector_path"])
    reset = raw.get("reset", {})
    aisbench = raw.get("aisbench", {})

    config = Config(
        services=services,
        cases=cases,
        modes=list(modes),
        block_tokens=block_tokens,
        work_dir=resolve(str(raw.get("work_dir", "work"))),
        results_dir=resolve(str(raw.get("results_dir", "results"))),
        configs_dir=(
            resolve(str(aisbench.get("configs_dir", "")))
            if aisbench.get("configs_dir")
            else ""
        ),
        summarizer=str(aisbench.get("summarizer", "default_perf")),
        aisbench_command=str(aisbench.get("command", "ais_bench")),
        aisbench_extra_args=[str(a) for a in aisbench.get("extra_args", [])],
        corpus_path=resolve(str(raw["corpus_path"])) if raw.get("corpus_path") else "",
        seed=raw.get("seed"),
        metrics_enabled=bool(metrics.get("enabled", False)),
        metrics_settle=float(metrics.get("settle_seconds", 1)),
        metrics_timeout=float(metrics.get("timeout_seconds", 10)),
        reset_timeout=float(reset.get("timeout_seconds", 60)),
        reset_interval=float(reset.get("retry_interval_seconds", 1)),
        source_path=os.path.abspath(path),
        raw=raw,
    )
    for name, value in (
        ("seed_settle_seconds", raw.get("seed_settle_seconds", 1)),
        ("metrics.settle_seconds", config.metrics_settle),
        ("metrics.timeout_seconds", config.metrics_timeout),
        ("reset.timeout_seconds", config.reset_timeout),
        ("reset.retry_interval_seconds", config.reset_interval),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ConfigError(f"{name} must be finite and non-negative")
    if config.metrics_timeout <= 0:
        raise ConfigError("metrics.timeout_seconds must be positive")
    return config
