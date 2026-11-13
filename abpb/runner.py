"""AISBench execution with the vllmbench 4801ebb benchmark workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

from . import aisbench, dataset, metrics
from .config import Case, ConfigError
from .metrics_capture import MetricsCapture
from .vllm_reporting import performance

RESET_EXTERNAL = {"cache": False, "posix-reset": True}
SHARED_MODES = ("hbm-shared", "hbm-shared-v4")
PARALLEL_MODES = ("hbm-shared", "cache")
DEFAULT_WARMUP = dict(
    enabled=True,
    rounds=1,
    input_len=8192,
    output_len=1,
    hit_rate=0.9,
    concurrency=4,
    num_requests=4,
)


@dataclass
class CaseResult:
    service: str
    case: str
    mode: str
    target_hit: float
    status: str = "complete"
    error: str = ""
    seed: int = 0
    perf: dict = field(default_factory=dict)
    hits: dict = field(default_factory=dict)
    reset: dict = field(default_factory=dict)
    started_at: str = ""
    details: dict = field(default_factory=dict)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _write_result(path, result):
    value = asdict(result)
    value.update(result.details)
    value.update(result.perf)
    value.update(result.hits)
    value["target_hit_rate"] = result.target_hit
    value["mode"] = result.mode.removesuffix("-run")
    _save(path, value)


def _run_id():
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + secrets.token_hex(4)
    )


def _check(perf, expected_requests, output_len, label, input_len=None):
    ok, failed = perf.get("requests_ok"), perf.get("requests_failed")
    if ok != expected_requests or failed != 0:
        raise aisbench.AisbenchError(
            f"{label}: expected {expected_requests} successes and zero failures, got {ok}/{failed}"
        )
    if perf.get("output_tokens") != expected_requests * output_len:
        raise aisbench.AisbenchError(
            f"{label}: output token count does not match {expected_requests} x {output_len}"
        )
    if (
        input_len is not None
        and perf.get("input_tokens") != expected_requests * input_len
    ):
        raise aisbench.AisbenchError(
            f"{label}: actual input tokens do not match {expected_requests} x {input_len}"
        )


def _measure(
    config,
    service,
    dataset_dir,
    output_dir,
    log_path,
    request_rate,
    output_len,
    concurrency,
    dp_rank=None,
    expect_requests=None,
    input_len=None,
):
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
        _check(
            run.metrics,
            expect_requests,
            output_len,
            os.path.basename(output_dir),
            input_len,
        )
    return run.metrics


def _ranks(service):
    return [service.dp_rank] if service.dp_rank is not None else list(range(service.dp))


def _seed(config, service, case, built, prefix_dir, out_dir, mode, log):
    if not built.prefix_rows:
        return
    ranks = (
        [service.dp_rank] if mode in ("posix-seed", "posix-reset") else _ranks(service)
    )
    shared = mode in SHARED_MODES
    seed_len = built.prefix_tokens + (1 if mode == "hbm-shared" else 0)

    def send(rank):
        label = "seed_auto" if rank is None else f"seed_dp{rank}"
        log(
            f"    {label} requests={len(built.prefix_rows)} concurrency={1 if shared else case.concurrency}"
        )
        return _measure(
            config,
            service,
            prefix_dir,
            os.path.join(out_dir, label),
            os.path.join(out_dir, label + ".log"),
            0,
            1,
            1 if shared else case.concurrency,
            rank,
            len(built.prefix_rows),
            seed_len,
        )

    if mode in PARALLEL_MODES and len(ranks) > 1:
        with ThreadPoolExecutor(max_workers=len(ranks)) as pool:
            list(pool.map(send, ranks))
    else:
        for rank in ranks:
            send(rank)
    time.sleep(
        max(float(config.raw.get("seed_settle_seconds", 1)), config.metrics_settle)
    )


def _warmup_settings(config):
    supplied = config.raw.get("ucm_warmup", config.raw.get("warmup", {}))
    if not isinstance(supplied, dict):
        raise ConfigError("warmup must be an object")
    settings = {
        **DEFAULT_WARMUP,
        **supplied,
    }
    if not isinstance(settings["enabled"], bool):
        raise ConfigError("warmup.enabled must be boolean")
    for name in ("input_len", "output_len", "concurrency", "num_requests", "rounds"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"warmup.{name} must be a positive integer")
    case = Case(
        **{
            key: settings[key]
            for key in (
                "input_len",
                "output_len",
                "hit_rate",
                "concurrency",
                "num_requests",
            )
        }
    )
    if not isinstance(settings["rounds"], int) or settings["rounds"] < 1:
        raise ConfigError("warmup.rounds must be a positive integer")
    if (
        min(case.input_len, case.output_len, case.concurrency, case.num_requests) < 1
        or case.concurrency > case.num_requests
        or not 0 <= case.hit_rate <= 1
    ):
        raise ConfigError("Invalid warmup case")
    return settings, case


def _fingerprint(config, service):
    value = dict(
        service=service.name,
        base_url=service.base_url,
        model=service.served_model,
        model_path=service.model_path,
        dp=service.dp,
        rank=service.dp_rank,
        cases=[asdict(case) for case in config.cases],
        block=config.block_tokens,
    )
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def _build_entry(tokenizer, corpus, picker, case, config, mode, seed, directory, name):
    built = dataset.build_dataset(
        tokenizer, corpus, case, config.block_tokens, seed, mode, picker
    )
    dataset.write_dataset_dir(str(directory / "prefix"), built.prefix_rows)
    dataset.write_dataset_dir(str(directory / "full"), built.full_rows)
    return dict(
        name=name,
        case=asdict(case),
        seed=seed,
        dataset_dir=str(directory),
        seed_tokens=built.prefix_tokens,
        expected_hbm_tokens=built.expected_hbm_tokens,
    )


def _load_entry(entry):
    root = Path(entry["dataset_dir"])

    def rows(name):
        return [
            json.loads(line)["question"]
            for line in (root / name / "test.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    case = Case(**entry["case"])
    built = dataset.CaseDataset(
        rows("prefix"),
        rows("full"),
        entry["seed_tokens"],
        case.input_len - entry["seed_tokens"],
        entry["seed"],
        entry.get("expected_hbm_tokens"),
    )
    return case, built


def _reset(config, service, mode, log):
    return metrics.reset_prefix_cache(
        service.base_url,
        RESET_EXTERNAL[mode],
        config.reset_timeout,
        config.reset_interval,
        log,
    )


def _warmup(config, service, entry, mode, directory, log):
    case, built = _load_entry(entry)
    if mode != "posix-run":
        _seed(
            config,
            service,
            case,
            built,
            str(Path(entry["dataset_dir"]) / "prefix"),
            str(directory),
            mode,
            log,
        )
    if mode == "posix-seed":
        return
    if mode in RESET_EXTERNAL:
        _reset(config, service, mode, log)
    for rank in _ranks(service):
        label = f"warmup_load_dp{rank}"
        _measure(
            config,
            service,
            str(Path(entry["dataset_dir"]) / "full"),
            str(directory / label),
            str(directory / (label + ".log")),
            0,
            case.output_len,
            case.concurrency,
            rank,
            case.num_requests,
            case.input_len,
        )
    time.sleep(
        max(float(config.raw.get("seed_settle_seconds", 1)), config.metrics_settle)
    )


def _perf_metrics(raw):
    native = dict(
        completed=raw.get("requests_ok"),
        failed=raw.get("requests_failed", 0),
        total_input_tokens=raw.get("input_tokens"),
        mean_ttft_ms=raw.get("ttft"),
        mean_tpot_ms=raw.get("tpot"),
        median_ttft_ms=raw.get("ttft_median"),
        p99_ttft_ms=raw.get("ttft_p99"),
        output_throughput=raw.get("output_throughput"),
        total_token_throughput=raw.get("total_throughput"),
        duration=raw.get("duration_ms", 0) / 1000,
    )
    # Missing AISBench duration is left missing, not replaced by CLI startup time.
    if "duration_ms" not in raw:
        native["duration"] = None
    if raw.get("requests_ok") is None:
        native.pop("completed")
    return {**raw, **performance(native)}


def run_all(config, log=print, dry_run=False):
    run_id = _run_id()
    results = []
    settings, warm_case = _warmup_settings(config)
    used = set()

    def seed_for(base, offset):
        if base is None or base == "random":
            value = secrets.randbelow(2**32)
            while value in used:
                value = secrets.randbelow(2**32)
        else:
            value = int(base) + offset
            if not 0 <= value < 2**32:
                raise ConfigError("Seed and case offsets must fit in 32 bits")
        used.add(value)
        return value

    for service in config.services:
        tokenizer = None
        corpus = []
        picker = None
        for mode_index, mode in enumerate(config.modes):
            mode_start = len(results)
            key = _fingerprint(config, service)
            pointer = Path(config.results_dir) / f".posix_pending_{key}.json"
            out = Path(config.results_dir) / run_id / service.name / mode
            work = Path(config.work_dir) / run_id / service.name / mode
            warmups = []
            entries = []
            manifest = dict(
                schema_version=2,
                mode=mode,
                fingerprint=key,
                status="running",
                service=service.name,
                run_id=run_id,
                cases=entries,
                warmups=warmups,
            )
            if dry_run:
                targets = (
                    [service.dp_rank]
                    if mode in ("posix-seed", "posix-reset")
                    else _ranks(service)
                )
                if mode in ("posix-run", "hbm-no-cache"):
                    targets = []
                log(
                    f"[dry-run] {service.name} {mode} seed_targets={targets} cases={len(config.cases)} warmup={settings['enabled'] and mode != 'hbm-no-cache'}"
                )
                continue
            try:
                metrics._get(service.base_url + "/health", config.metrics_timeout)
                if mode == "posix-run":
                    source = config.raw.get("seed_run")
                    source = (
                        Path(source)
                        if source
                        else Path(json.loads(pointer.read_text())["manifest"])
                    )
                    if source.is_dir():
                        source = source / "manifest.json"
                    saved = json.loads(source.read_text(encoding="utf-8"))
                    if saved["fingerprint"] != key or saved["status"] != "seeded":
                        raise ConfigError(
                            "Posix seed config differs or seeding did not complete"
                        )
                    entries.extend(saved["cases"])
                    if settings["enabled"]:
                        warmups.extend(saved["warmups"])
                    manifest["seed_source"] = str(source)
                else:
                    if tokenizer is None:
                        tokenizer = dataset.load_tokenizer(service.tokenizer_path)
                        corpus = dataset.load_corpus(config.corpus_path)
                        history_key = hashlib.sha256(
                            (service.model_path + config.corpus_path).encode()
                        ).hexdigest()[:16]
                        picker = dataset.PrefixPicker(
                            corpus,
                            Path(config.work_dir) / f"picked_ids_{history_key}.json",
                        )
                    offset = mode_index * (len(config.cases) + settings["rounds"])
                    for index, case in enumerate(config.cases):
                        name = (
                            f"{index+1:03d}_{case.input_len}in_{case.output_len}out"
                            f"_c{case.concurrency}_n{case.num_requests}"
                        )
                        value = seed_for(
                            case.seed if case.seed is not None else config.seed,
                            offset + index,
                        )
                        entries.append(
                            _build_entry(
                                tokenizer,
                                corpus,
                                picker,
                                case,
                                config,
                                mode,
                                value,
                                work / name,
                                name,
                            )
                        )
                    if settings["enabled"] and mode != "hbm-no-cache":
                        for index in range(settings["rounds"]):
                            name = f"warmup_{index+1:03d}"
                            value = seed_for(
                                config.seed, offset + len(config.cases) + index
                            )
                            warmups.append(
                                _build_entry(
                                    tokenizer,
                                    corpus,
                                    picker,
                                    warm_case,
                                    config,
                                    mode,
                                    value,
                                    work / name,
                                    name,
                                )
                            )
                _save(out / "manifest.json", manifest)
                if (
                    settings["enabled"]
                    and mode in ("cache", "posix-seed", "posix-reset")
                    and warm_case.prefix_tokens(config.block_tokens, mode) == 0
                ):
                    raise ConfigError(
                        "Warmup requires at least one cached block to exercise loading"
                    )
                for entry in warmups:
                    _warmup(config, service, entry, mode, work / entry["name"], log)
                for entry in entries:
                    case, built = _load_entry(entry)
                    directory = out / entry["name"]
                    raw_dir = work / entry["name"]
                    result = CaseResult(
                        service.name,
                        entry["name"],
                        mode,
                        0 if mode == "hbm-no-cache" else case.hit_rate,
                        seed=entry["seed"],
                        started_at=datetime.now(timezone.utc).isoformat(),
                    )
                    result.details = dict(
                        run_id=run_id,
                        served_model=service.served_model,
                        dp=service.dp,
                        routing=(
                            "auto"
                            if service.dp_rank is None
                            else f"fixed_dp{service.dp_rank}"
                        ),
                        input_len=case.input_len,
                        output_len=case.output_len,
                        concurrency=case.concurrency,
                        num_requests=case.num_requests,
                        seed_tokens=built.prefix_tokens,
                        expected_hbm_tokens=built.expected_hbm_tokens,
                    )
                    try:
                        if mode not in ("posix-run", "hbm-no-cache"):
                            _seed(
                                config,
                                service,
                                case,
                                built,
                                str(Path(entry["dataset_dir"]) / "prefix"),
                                str(raw_dir),
                                mode,
                                log,
                            )
                        if mode == "posix-seed":
                            continue
                        if mode in RESET_EXTERNAL:
                            result.reset = _reset(config, service, mode, log)
                        capture = MetricsCapture(
                            directory / "metrics",
                            dict(
                                enabled=config.metrics_enabled,
                                url=service.base_url + "/metrics",
                                settle_seconds=config.metrics_settle,
                                timeout_seconds=config.metrics_timeout,
                                collector_path=config.raw.get("metrics", {}).get(
                                    "collector_path"
                                ),
                            ),
                        )
                        captured = False
                        try:
                            capture.start()
                            captured = True
                        except Exception as exc:
                            result.hits["metrics_status"] = (
                                f"hit counters unavailable: {exc}"
                            )
                            if config.metrics_enabled:
                                raise
                        try:
                            log(
                                f"    [{mode}] measure {entry['name']} seed={entry['seed']}"
                            )
                            raw = _measure(
                                config,
                                service,
                                str(Path(entry["dataset_dir"]) / "full"),
                                str(raw_dir / "measure"),
                                str(raw_dir / "measure.log"),
                                case.request_rate,
                                case.output_len,
                                case.concurrency,
                                service.dp_rank,
                                case.num_requests,
                                case.input_len,
                            )
                            result.perf = _perf_metrics(raw)
                        finally:
                            if captured:
                                try:
                                    result.hits.update(
                                        capture.finish(
                                            {service.served_model, service.model_path}
                                        )
                                    )
                                except Exception as exc:
                                    result.hits["metrics_status"] = (
                                        f"hit counters unavailable: {exc}"
                                    )
                                    if config.metrics_enabled:
                                        raise
                    except Exception as exc:
                        result.status, result.error = (
                            "error",
                            f"{type(exc).__name__}: {exc}",
                        )
                        raise
                    finally:
                        if mode != "posix-seed":
                            _write_result(directory / "case_result.json", result)
                            results.append(result)
                manifest["status"] = "seeded" if mode == "posix-seed" else "complete"
                _save(out / "manifest.json", manifest)
                if mode == "posix-seed":
                    _save(pointer, dict(manifest=str(out / "manifest.json")))
                    log(
                        "Posix seeded. Wait for writes to finish, stop/restart the service, then run posix-run."
                    )
            except Exception as exc:
                manifest["status"], manifest["error"] = (
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                )
                _save(out / "manifest.json", manifest)
                log(manifest["error"])
                if len(results) == mode_start or results[-1].status != "error":
                    error = CaseResult(
                        service.name,
                        "workflow",
                        mode,
                        0,
                        status="error",
                        error=manifest["error"],
                    )
                    error.details = dict(
                        run_id=run_id, served_model=service.served_model
                    )
                    _write_result(out / "workflow" / "case_result.json", error)
                    results.append(error)
            finally:
                if mode != "posix-seed":
                    from . import reporting

                    reporting.write_reports(Path(config.results_dir))
    return results
