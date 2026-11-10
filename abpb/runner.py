"""Workflow: warm the load path, seed prefixes, clear cache, measure."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field

from . import aisbench, dataset, metrics
from .config import Case

RESET_EXTERNAL = {"cache": False, "posix-reset": True}

DEFAULT_WARMUP = {
    "enabled": True,
    "input_len": 8192,
    "output_len": 1,
    "hit_rate": 0.9,
    "concurrency": 4,
    "num_requests": 4,
}


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
    dp_rank: int | None = None,
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
        dp_rank=dp_rank,
    )
    return run.metrics


def _seed(config, service, built, prefix_dir: str, out_dir: str, log) -> None:
    """Plant each distinct prefix on every DP rank (这是预埋，不是预热)."""
    log(f"    seed {len(built.prefix_rows)} prefix(es) on {service.dp} DP")
    for rank in range(max(service.dp, 1)):
        _measure(
            config, service, prefix_dir,
            os.path.join(out_dir, f"dp{rank}"),
            os.path.join(out_dir, f"dp{rank}.log"),
            request_rate=0,
            output_len=1,
            concurrency=max(1, len(built.prefix_rows)),
            dp_rank=rank,
        )
    time.sleep(float(config.raw.get("seed_settle_seconds", 10)))


def _warmup_mode(config, service, tokenizer, corpus, mode: str, run_dir: str, log) -> None:
    """Run one full mini-cycle so the load path (not just the request path) is warm."""
    settings = {**DEFAULT_WARMUP, **(config.raw.get("warmup") or {})}
    if not settings.get("enabled", True):
        return
    case = Case(
        input_len=int(settings["input_len"]),
        output_len=int(settings["output_len"]),
        hit_rate=float(settings["hit_rate"]),
        concurrency=int(settings["concurrency"]),
        num_requests=int(settings["num_requests"]),
    )
    built = dataset.build_dataset(tokenizer, corpus, case, config.block_tokens, seed=987654)
    warm_dir = os.path.join(run_dir, service.name, f"_warmup_{mode}")
    dataset.write_dataset_dir(os.path.join(warm_dir, "prefix"), built.prefix_rows)
    dataset.write_dataset_dir(os.path.join(warm_dir, "full"), built.full_rows)
    log(
        f"  warmup[{mode}] {case.num_requests}x{case.input_len} "
        f"hit={built.prefix_tokens / case.input_len:.2f} out={case.output_len}"
    )
    try:
        _seed(config, service, built, os.path.join(warm_dir, "prefix"),
              os.path.join(warm_dir, "seed"), log)
        if mode in RESET_EXTERNAL:
            metrics.reset_prefix_cache(
                service.base_url,
                reset_external=RESET_EXTERNAL[mode],
                timeout_s=config.reset_timeout,
                interval_s=config.reset_interval,
                log=log,
            )
        for rank in range(max(service.dp, 1)):
            _measure(
                config, service, os.path.join(warm_dir, "full"),
                os.path.join(warm_dir, f"load_dp{rank}"),
                os.path.join(warm_dir, f"load_dp{rank}.log"),
                request_rate=0,
                output_len=case.output_len,
                concurrency=case.concurrency,
                dp_rank=rank,
            )
    except Exception as exc:  # noqa: BLE001 - warmup must not abort the matrix
        log(f"  warmup[{mode}] failed (continuing): {type(exc).__name__}: {exc}")


def run_all(config, log=print, dry_run: bool = False) -> list[CaseResult]:
    run_id = _run_id()
    run_dir = os.path.join(config.work_dir, run_id)
    results: list[CaseResult] = []
    corpus = dataset.load_corpus(config.corpus_path) if not dry_run else []

    for service in config.services:
        if dry_run:
            for case in config.cases:
                log(f"[dry-run] {service.name} {case.label(config.block_tokens)}")
            continue

        tokenizer = dataset.load_tokenizer(service.tokenizer_path)
        prepared = []
        for case in config.cases:
            seed = case.seed if case.seed is not None else (
                config.seed if config.seed is not None else random.randrange(1 << 30)
            )
            built = dataset.build_dataset(
                tokenizer, corpus, case, config.block_tokens, seed
            )
            case_dir = os.path.join(run_dir, service.name, case.label(config.block_tokens))
            dataset.write_dataset_dir(os.path.join(case_dir, "dataset_prefix"), built.prefix_rows)
            dataset.write_dataset_dir(os.path.join(case_dir, "dataset_full"), built.full_rows)
            log(
                f"{service.name} {case.label(config.block_tokens)} seed={seed} "
                f"prefix={built.prefix_tokens}tok suffix={built.suffix_tokens}tok "
                f"prefixes={len(built.prefix_rows)}"
            )
            prepared.append((case, built, case_dir))

        for mode in config.modes:
            _warmup_mode(config, service, tokenizer, corpus, mode, run_dir, log)
            for case, built, case_dir in prepared:
                mode_dir = os.path.join(case_dir, mode)
                result = CaseResult(
                    service=service.name,
                    case=case.label(config.block_tokens),
                    mode=mode,
                    target_hit=built.prefix_tokens / case.input_len,
                    seed=built.seed,
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                try:
                    _seed(config, service, built,
                          os.path.join(case_dir, "dataset_prefix"),
                          os.path.join(mode_dir, "seed"), log)
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

                    log(f"    [{mode}] measure")
                    result.perf = _measure(
                        config, service, os.path.join(case_dir, "dataset_full"),
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
                    log(f"    [{mode}] failed: {result.error}")
                _write_result(os.path.join(mode_dir, "case_result.json"), result)
                results.append(result)
    return results
