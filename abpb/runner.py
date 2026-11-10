"""Workflow: warm the load path, seed prefixes, clear cache, measure."""

from __future__ import annotations

import json
import os
import random
import time
import zlib
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


def _mode_seed(base: int, mode: str) -> int:
    """Give every mode its own keys so one mode's data cannot seed the next."""
    return (base ^ zlib.crc32(mode.encode("utf-8"))) & 0x7FFFFFFF


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


def _check(perf: dict, expected_requests: int, output_len: int, label: str) -> None:
    """A run that silently dropped requests must not be reported as ok."""
    ok = perf.get("requests_ok")
    failed = perf.get("requests_failed")
    if ok is None or failed is None:
        raise aisbench.AisbenchError(f"{label}: perf result has no request counts")
    if failed > 0:
        raise aisbench.AisbenchError(f"{label}: {int(failed)} request(s) failed")
    if int(ok) != expected_requests:
        raise aisbench.AisbenchError(
            f"{label}: {int(ok)} request(s) completed, expected {expected_requests}"
        )
    if output_len > 1:
        produced = perf.get("output_tokens") or 0
        if produced < ok * output_len * 0.9:
            raise aisbench.AisbenchError(
                f"{label}: {int(produced)} output tokens, expected about {int(ok * output_len)}"
            )


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
    expect_requests: int | None = None,
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
    if expect_requests is not None:
        _check(run.metrics, expect_requests, output_len, os.path.basename(output_dir))
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
            expect_requests=len(built.prefix_rows),
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
    built = dataset.build_dataset(
        tokenizer, corpus, case, config.block_tokens,
        seed=_mode_seed(987654, mode),
    )
    warm_dir = os.path.join(run_dir, service.name, f"_warmup_{mode}")
    dataset.write_dataset_dir(os.path.join(warm_dir, "prefix"), built.prefix_rows)
    dataset.write_dataset_dir(os.path.join(warm_dir, "full"), built.full_rows)
    log(
        f"  warmup[{mode}] {case.num_requests}x{case.input_len} "
        f"hit={built.prefix_tokens / case.input_len:.2f} out={case.output_len}"
    )
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
            expect_requests=case.num_requests,
        )


def run_all(config, log=print, dry_run: bool = False) -> list[CaseResult]:
    run_id = _run_id()
    run_dir = os.path.join(config.work_dir, run_id)
    results: list[CaseResult] = []
    corpus = dataset.load_corpus(config.corpus_path) if not dry_run else []

    for service in config.services:
        if dry_run:
            for case in config.cases:
                log(f"[dry-run] {service.name} {case.label(config.block_tokens)} "
                    f"x {len(config.modes)} mode(s)")
            continue

        tokenizer = dataset.load_tokenizer(service.tokenizer_path)
        base_seeds = {}
        for case in config.cases:
            base_seeds[case.label(config.block_tokens)] = (
                case.seed if case.seed is not None
                else config.seed if config.seed is not None
                else random.randrange(1 << 30)
            )

        for mode in config.modes:
            try:
                _warmup_mode(config, service, tokenizer, corpus, mode, run_dir, log)
            except Exception as exc:  # noqa: BLE001 - a failed warmup invalidates the mode
                log(f"  warmup[{mode}] failed, skipping this mode: {type(exc).__name__}: {exc}")
                policy = config.raw.get("warmup_on_failure", "skip")
                if policy != "continue":
                    for case in config.cases:
                        label = case.label(config.block_tokens)
                        result = CaseResult(
                            service=service.name, case=label, mode=mode,
                            target_hit=case.prefix_tokens(config.block_tokens) / case.input_len,
                            status="error",
                            error=f"warmup failed: {type(exc).__name__}: {exc}",
                            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        _write_result(
                            os.path.join(run_dir, service.name, label, mode, "case_result.json"),
                            result,
                        )
                        results.append(result)
                    continue

            for case in config.cases:
                label = case.label(config.block_tokens)
                seed = _mode_seed(base_seeds[label], mode)
                built = dataset.build_dataset(
                    tokenizer, corpus, case, config.block_tokens, seed
                )
                case_dir = os.path.join(run_dir, service.name, label, mode)
                dataset.write_dataset_dir(os.path.join(case_dir, "dataset_prefix"), built.prefix_rows)
                dataset.write_dataset_dir(os.path.join(case_dir, "dataset_full"), built.full_rows)
                log(
                    f"  [{mode}] {label} seed={seed} prefix={built.prefix_tokens}tok "
                    f"suffix={built.suffix_tokens}tok prefixes={len(built.prefix_rows)}"
                )

                result = CaseResult(
                    service=service.name,
                    case=label,
                    mode=mode,
                    target_hit=built.prefix_tokens / case.input_len,
                    seed=seed,
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                try:
                    _seed(config, service, built,
                          os.path.join(case_dir, "dataset_prefix"),
                          os.path.join(case_dir, "seed"), log)
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
                        before = metrics.scrape(
                            service.base_url, config.metrics_timeout, service.served_model
                        )

                    log("    measure")
                    result.perf = _measure(
                        config, service, os.path.join(case_dir, "dataset_full"),
                        os.path.join(case_dir, "measure"),
                        os.path.join(case_dir, "measure.log"),
                        request_rate=float(case.request_rate),
                        output_len=case.output_len,
                        concurrency=case.concurrency,
                        expect_requests=case.num_requests,
                    )
                    if config.metrics_enabled:
                        time.sleep(config.metrics_settle)
                        after = metrics.scrape(
                            service.base_url, config.metrics_timeout, service.served_model
                        )
                        result.hits = metrics.hit_summary(before, after)
                        if result.hits.get("counter_reset"):
                            log("    metrics counters reset inside the window; hit rate dropped")
                except Exception as exc:  # noqa: BLE001 - one failing cell must not stop the matrix
                    result.status = "error"
                    result.error = f"{type(exc).__name__}: {exc}"
                    log(f"    failed: {result.error}")
                _write_result(os.path.join(case_dir, "case_result.json"), result)
                results.append(result)
    return results
