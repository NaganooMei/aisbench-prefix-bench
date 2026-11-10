"""Workflow: seed prefixes -> clear cache -> measure with AISBench."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field

from . import aisbench, dataset, metrics
from .config import Case

RESET_EXTERNAL = {"cache": False, "posix-reset": True}


@dataclass
class CaseResult:
    service: str
    case: str
    mode: str
    target_hit: float
    status: str = "ok"
    error: str = ""
    seed: int = 0
    perf: dict = field(default_factory=dict)
    hits: dict = field(default_factory=dict)
    reset: dict = field(default_factory=dict)
    started_at: str = ""


def _run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _write_result(path: str, result: CaseResult) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "service": result.service,
        "case": result.case,
        "mode": result.mode,
        "target_hit": result.target_hit,
        "status": result.status,
        "error": result.error,
        "seed": result.seed,
        "started_at": result.started_at,
        "perf": result.perf,
        "hits": result.hits,
        "reset": result.reset,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _measure(
    config,
    service,
    dataset_dir: str,
    output_dir: str,
    log_path: str,
    request_rate: float,
    output_len: int,
    concurrency: int,
) -> dict:
    run = aisbench.execute(
        config,
        service,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        log_path=log_path,
        request_rate=request_rate,
        output_len=output_len,
        concurrency=concurrency,
    )
    return run.metrics


def _warmup(config, service, tokenizer, corpus, run_dir: str, log) -> None:
    """Absorb first-run costs (graph capture, batch shape) outside the matrix."""
    settings = config.raw.get("warmup") or {}
    template = config.cases[0]
    case = Case(
        input_len=int(settings.get("input_len", template.input_len)),
        output_len=int(settings.get("output_len", template.output_len)),
        hit_rate=float(settings.get("hit_rate", template.hit_rate)),
        concurrency=int(settings.get("concurrency", template.concurrency)),
        num_requests=int(settings.get("num_requests", template.num_requests)),
    )
    warm_dir = os.path.join(run_dir, service.name, "_warmup")
    dataset_dir = os.path.join(warm_dir, "dataset")
    built = dataset.build_dataset(
        tokenizer, corpus, case, config.block_tokens, seed=12345, dp=1
    )
    dataset.write_dataset_dir(dataset_dir, built.full_rows)
    log(f"{service.name} warmup ({case.num_requests} requests)")
    try:
        _measure(
            config, service, dataset_dir,
            os.path.join(warm_dir, "run"),
            os.path.join(warm_dir, "warmup.log"),
            request_rate=0,
            output_len=case.output_len,
            concurrency=case.concurrency,
        )
    except Exception as exc:  # noqa: BLE001 - warmup must not abort the matrix
        log(f"  warmup failed (continuing): {type(exc).__name__}: {exc}")


def run_all(config, log=print, dry_run: bool = False) -> list[CaseResult]:
    run_id = _run_id()
    run_dir = os.path.join(config.work_dir, run_id)
    results: list[CaseResult] = []
    corpus = dataset.load_corpus(config.corpus_path) if not dry_run else []

    for service in config.services:
        tokenizer = None if dry_run else dataset.load_tokenizer(service.tokenizer_path)
        if not dry_run and (config.raw.get("warmup") or {}).get("enabled", True):
            _warmup(config, service, tokenizer, corpus, run_dir, log)
        for case in config.cases:
            case_label = case.label(config.block_tokens)
            seed = case.seed if case.seed is not None else (
                config.seed if config.seed is not None else random.randrange(1 << 30)
            )
            case_dir = os.path.join(run_dir, service.name, case_label)
            if dry_run:
                log(f"[dry-run] {service.name} {case_label} seed={seed}")
                for mode in config.modes:
                    log(f"  would run {mode}")
                continue

            built = dataset.build_dataset(
                tokenizer, corpus, case, config.block_tokens, seed, service.dp
            )
            prefix_dir = os.path.join(case_dir, "dataset_prefix")
            full_dir = os.path.join(case_dir, "dataset_full")
            dataset.write_dataset_dir(prefix_dir, built.prefix_rows)
            dataset.write_dataset_dir(full_dir, built.full_rows)
            log(
                f"{service.name} {case_label} seed={seed} "
                f"prefix={built.prefix_tokens}tok suffix={built.suffix_tokens}tok "
                f"prefixes={case.prefix_num} seed_rows={len(built.prefix_rows)}"
            )

            for mode in config.modes:
                mode_dir = os.path.join(case_dir, mode)
                result = CaseResult(
                    service=service.name,
                    case=case_label,
                    mode=mode,
                    target_hit=built.prefix_tokens / case.input_len,
                    seed=seed,
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                try:
                    log(f"  [{mode}] seeding prefixes")
                    _measure(
                        config, service, prefix_dir,
                        os.path.join(mode_dir, "seed"),
                        os.path.join(mode_dir, "seed.log"),
                        request_rate=0,
                        output_len=1,
                        concurrency=max(1, len(built.prefix_rows)),
                    )
                    settle = float(config.raw.get("seed_settle_seconds", 10))
                    time.sleep(settle)

                    if mode in RESET_EXTERNAL:
                        result.reset = metrics.reset_prefix_cache(
                            service.base_url,
                            reset_external=RESET_EXTERNAL[mode],
                            timeout_s=config.reset_timeout,
                            interval_s=config.reset_interval,
                            log=log,
                        )

                    before = {}
                    if config.metrics_enabled:
                        time.sleep(config.metrics_settle)
                        before = metrics.scrape(service.base_url, config.metrics_timeout)

                    log(f"  [{mode}] measuring")
                    result.perf = _measure(
                        config, service, full_dir,
                        os.path.join(mode_dir, "measure"),
                        os.path.join(mode_dir, "measure.log"),
                        request_rate=float(case.request_rate),
                        output_len=case.output_len,
                        concurrency=case.concurrency,
                    )
                    if config.metrics_enabled:
                        time.sleep(config.metrics_settle)
                        after = metrics.scrape(service.base_url, config.metrics_timeout)
                        result.hits = metrics.hit_summary(before, after)
                except Exception as exc:  # noqa: BLE001 - one failing cell must not stop the matrix
                    result.status = "error"
                    result.error = f"{type(exc).__name__}: {exc}"
                    log(f"  [{mode}] failed: {result.error}")
                _write_result(os.path.join(mode_dir, "case_result.json"), result)
                results.append(result)
    return results
