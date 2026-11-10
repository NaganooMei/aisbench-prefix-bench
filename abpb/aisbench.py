"""AISBench glue: write config files, run perf mode, parse its results."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlparse

MODEL_ABBR = "abpb_model"
DATASET_ABBR = "abpb_gsm8k_perf"

MODEL_TEMPLATE = '''\
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="{abbr}",
        path="{tokenizer}",
        model="{served_model}",
        api_key="{api_key}",
        request_rate={request_rate},
        retry=2,
        host_ip="{host}",
        host_port={port},
        url="",
        max_out_len={output_len},
        batch_size={concurrency},
        trust_remote_code=True,
        generation_kwargs=dict(temperature=0, ignore_eos=True),
    )
]
'''

DATASET_TEMPLATE = '''\
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
        abbr='{abbr}',
        type=GSM8KDataset,
        path='{dataset_dir}',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
'''


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
) -> tuple[str, str]:
    parsed = urlparse(service.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    model_path = os.path.join(configs_dir, "models", "vllm_api", f"{MODEL_ABBR}.py")
    dataset_path = os.path.join(configs_dir, "datasets", "gsm8k", f"{DATASET_ABBR}.py")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    with open(model_path, "w", encoding="utf-8") as fh:
        fh.write(
            MODEL_TEMPLATE.format(
                abbr=MODEL_ABBR,
                tokenizer=service.tokenizer_path,
                served_model=service.served_model,
                api_key=service.api_key,
                request_rate=request_rate,
                host=host,
                port=port,
                output_len=output_len,
                concurrency=concurrency,
            )
        )
    with open(dataset_path, "w", encoding="utf-8") as fh:
        fh.write(DATASET_TEMPLATE.format(abbr=DATASET_ABBR, dataset_dir=dataset_dir))
    return model_path, dataset_path


def build_command(config, output_dir: str) -> list[str]:
    cmd = [
        config.aisbench_command,
        "--models", MODEL_ABBR,
        "--datasets", DATASET_ABBR,
        "--mode", "perf",
        "--summarizer", config.summarizer,
        "--work-dir", output_dir,
        "--num-warmups", "0",
    ]
    cmd.extend(config.aisbench_extra_args)
    return cmd


def run(cmd: list[str], log_path: str, timeout_s: float) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
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


THROUGHPUT_KEYS = {
    "ttft": ("ttft", "mean_ttft_ms"),
    "ttft_p90": ("ttft_p90", "p90_ttft_ms"),
    "tpot": ("tpot", "mean_tpot_ms"),
    "e2e": ("e2e", "e2el", "mean_e2el_ms"),
    "output_throughput": ("output_token_throughput", "output_throughput"),
    "total_throughput": ("total_token_throughput",),
    "request_throughput": ("request_throughput_rps", "request_throughput"),
}


def _match_metric(payload: dict, names: tuple[str, ...]) -> float | None:
    for key, value in payload.items():
        lower = key.lower()
        if any(name in lower for name in names):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def parse_results(output_dir: str) -> dict:
    """Read the per-dataset perf JSON AISBench writes under <work-dir>."""
    candidates = []
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if name.endswith(".json") and DATASET_ABBR in name:
                candidates.append(os.path.join(root, name))
    if not candidates:
        return {}
    newest = max(candidates, key=os.path.getmtime)
    with open(newest, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        return {}
    metrics = {}
    for field, names in THROUGHPUT_KEYS.items():
        value = _match_metric(payload, names)
        if value is not None:
            metrics[field] = value
    metrics["source"] = newest
    return metrics


STDOUT_RULES = (
    ("ttft", r"TTFT[^0-9]*([0-9]+\.[0-9]+)", 1.0),
    ("tpot", r"TPOT[^0-9]*([0-9]+\.[0-9]+)", 1.0),
    ("output_throughput", r"Output Token Throughput[^0-9]*([0-9]+\.[0-9]+)", 1.0),
    ("total_throughput", r"Total Token Throughput[^0-9]*([0-9]+\.[0-9]+)", 1.0),
)


def parse_stdout(stdout: str) -> dict:
    """Fallback for AISBench versions that only print a summary table."""
    metrics = {}
    for field, pattern, scale in STDOUT_RULES:
        match = re.search(pattern, stdout)
        if match:
            metrics[field] = float(match.group(1)) * scale
    return metrics


def execute(config, service, dataset_dir: str, output_dir: str, log_path: str,
            request_rate: float, output_len: int, concurrency: int) -> RunResult:
    configs_dir = config.configs_dir or find_configs_dir()
    write_configs(configs_dir, service, dataset_dir, request_rate, output_len, concurrency)
    cmd = build_command(config, output_dir)
    started = time.time()
    stdout = run(cmd, log_path, float(config.raw.get("aisbench", {}).get("timeout_seconds", 3600)))
    metrics = parse_results(output_dir) or parse_stdout(stdout)
    metrics["wall_seconds"] = round(time.time() - started, 1)
    return RunResult(metrics=metrics, stdout=stdout, output_dir=output_dir, command=cmd)
