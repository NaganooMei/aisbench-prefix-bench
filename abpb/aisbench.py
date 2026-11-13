"""AISBench glue: write config files, run perf mode, parse its results."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import secrets
from dataclasses import dataclass
from urllib.parse import urlparse


def _abbrs() -> tuple[str, str]:
    """Unique config names per process: two runs share one AISBench install."""
    token = f"{os.getpid()}_{secrets.token_hex(6)}"
    return f"abpb_model_{token}", f"abpb_gsm8k_perf_{token}"


MODEL_ABBR, DATASET_ABBR = _abbrs()

MODEL_TEMPLATE = """\
from abpb.aisbench_model import RankPinnedStreamCompletion

models = [
    dict(
        attr="service",
        type=RankPinnedStreamCompletion,
        dp_rank={dp_rank},
        abbr={abbr!r},
        path={tokenizer!r},
        model={served_model!r},
        api_key={api_key!r},
        request_rate={request_rate},
        retry=1,
        host_ip={host!r},
        host_port={port},
        url={base_url!r},
        max_out_len={output_len},
        batch_size={concurrency},
        trust_remote_code=True,
        generation_kwargs=dict(temperature=0, ignore_eos=True, add_special_tokens=False),
    )
]
"""

DATASET_TEMPLATE = """\
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import GSM8KDataset

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{{question}}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gsm8k_eval_cfg = dict(pred_role='BOT')

gsm8k_datasets = [
    dict(
        abbr={abbr!r},
        type=GSM8KDataset,
        path={dataset_dir!r},
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
"""


class AisbenchError(RuntimeError):
    pass


def find_configs_dir() -> str:
    """Directory that holds AISBench's shipped `configs/` tree."""
    code = (
        "import os, ais_bench;"
        "print(os.path.join(os.path.dirname(ais_bench.__file__), 'benchmark', 'configs'))"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AisbenchError(f"cannot run python to locate ais_bench: {exc}") from exc
    path = out.stdout.strip()
    if out.returncode != 0 or not path or not os.path.isdir(path):
        raise AisbenchError(
            "ais_bench is not importable; install it in this environment "
            f"(stderr: {out.stderr.strip()[:200]})"
        )
    return path


@dataclass
class RunResult:
    metrics: dict
    stdout: str
    output_dir: str
    command: list[str]


def write_configs(
    configs_dir: str,
    service,
    dataset_dir: str,
    request_rate: float,
    output_len: int,
    concurrency: int,
    dp_rank: int | None = None,
    model_abbr: str = MODEL_ABBR,
    dataset_abbr: str = DATASET_ABBR,
) -> tuple[str, str]:
    parsed = urlparse(service.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    model_path = os.path.join(configs_dir, "models", "vllm_api", f"{model_abbr}.py")
    dataset_path = os.path.join(configs_dir, "datasets", "gsm8k", f"{dataset_abbr}.py")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    with open(model_path, "w", encoding="utf-8") as fh:
        fh.write(
            MODEL_TEMPLATE.format(
                abbr=model_abbr,
                tokenizer=service.tokenizer_path,
                served_model=service.served_model,
                api_key=service.api_key,
                request_rate=request_rate,
                host=host,
                port=port,
                output_len=output_len,
                concurrency=concurrency,
                dp_rank="None" if dp_rank is None else int(dp_rank),
                base_url=service.base_url.rstrip("/") + "/",
            )
        )
    with open(dataset_path, "w", encoding="utf-8") as fh:
        fh.write(DATASET_TEMPLATE.format(abbr=dataset_abbr, dataset_dir=dataset_dir))
    return model_path, dataset_path


def build_command(
    config, output_dir: str, model_abbr=MODEL_ABBR, dataset_abbr=DATASET_ABBR
) -> list[str]:
    cmd = [
        config.aisbench_command,
        "--models",
        model_abbr,
        "--datasets",
        dataset_abbr,
        "--mode",
        "perf",
        "--summarizer",
        config.summarizer,
        "--work-dir",
        output_dir,
        "--num-warmups",
        "0",
    ]
    cmd.extend(config.aisbench_extra_args)
    return cmd


def run(cmd: list[str], log_path: str, timeout_s: float) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, env=env
            )
        except subprocess.TimeoutExpired as exc:
            raise AisbenchError(f"ais_bench timed out after {timeout_s:.0f}s") from exc
        except OSError as exc:
            raise AisbenchError(f"cannot launch ais_bench: {exc}") from exc
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        stdout = fh.read()
    if proc.returncode != 0:
        tail = "\n".join(stdout.splitlines()[-15:])
        raise AisbenchError(f"ais_bench exited {proc.returncode}:\n{tail}")
    return stdout


JSON_METRICS = {
    "benchmark duration": "duration_ms",
    "request throughput": "request_throughput",
    "input token throughput": "input_throughput",
    "output token throughput": "output_throughput",
    "total token throughput": "total_throughput",
    "total requests": "requests",
    "success requests": "requests_ok",
    "failed requests": "requests_failed",
    "total input tokens": "input_tokens",
    "total generated tokens": "output_tokens",
}

_NUMBER = re.compile(r"[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")


def _number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        match = _NUMBER.search(value.replace(",", ""))
        if match:
            return float(match.group(0))
    return None


def parse_results(output_dir: str, dataset_abbr=DATASET_ABBR) -> dict:
    """Read the per-dataset perf JSON AISBench writes under <work-dir>."""
    candidates = []
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if (
                name.endswith(".json")
                and dataset_abbr in name
                and "details" not in name
            ):
                candidates.append(os.path.join(root, name))
    if not candidates:
        return {}
    newest = max(candidates, key=os.path.getmtime)
    with open(newest, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    metrics = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            field = JSON_METRICS.get(key.strip().lower())
            if not field:
                continue
            raw = value.get("total") if isinstance(value, dict) else value
            number = _number(raw)
            if number is not None:
                metrics[field] = number
    metrics["source"] = newest
    return metrics


# "│ TTFT │ total │ avg │ min │ max │ median │ P75 │ P90 │ P99 │ N │"
TABLE_ROWS = {
    "TTFT": "ttft",
    "TPOT": "tpot",
    "ITL": "itl",
    "E2EL": "e2el",
}


def parse_stdout(stdout: str) -> dict:
    """Extract Average / P90 from AISBench's performance results table."""
    metrics = {}
    for line in stdout.splitlines():
        if not line.startswith("│"):
            continue
        cells = [cell.strip() for cell in line.strip("│").split("│")]
        if len(cells) < 9:
            continue
        name = cells[0]
        field = TABLE_ROWS.get(name)
        if not field or cells[1] != "total":
            continue
        average = _number(cells[2])
        p90 = _number(cells[7])
        if average is not None:
            metrics[field] = average
        if p90 is not None:
            metrics[f"{field}_p90"] = p90
        for suffix, index in (("median", 5), ("p99", 8)):
            number = _number(cells[index])
            if number is not None:
                metrics[f"{field}_{suffix}"] = number
    return metrics


def execute(
    config,
    service,
    dataset_dir: str,
    output_dir: str,
    log_path: str,
    request_rate: float,
    output_len: int,
    concurrency: int,
    dp_rank: int | None = None,
) -> RunResult:
    configs_dir = config.configs_dir or find_configs_dir()
    model_abbr, dataset_abbr = _abbrs()
    paths = write_configs(
        configs_dir,
        service,
        dataset_dir,
        request_rate,
        output_len,
        concurrency,
        dp_rank,
        model_abbr,
        dataset_abbr,
    )
    cmd = build_command(config, output_dir, model_abbr, dataset_abbr)
    started = time.time()
    try:
        stdout = run(
            cmd,
            log_path,
            float(config.raw.get("aisbench", {}).get("timeout_seconds", 3600)),
        )
        metrics = parse_results(output_dir, dataset_abbr)
    finally:
        for path in paths:
            os.unlink(path)
    for key, value in parse_stdout(stdout).items():
        metrics.setdefault(key, value)
    metrics["wall_seconds"] = round(time.time() - started, 1)
    return RunResult(metrics=metrics, stdout=stdout, output_dir=output_dir, command=cmd)
