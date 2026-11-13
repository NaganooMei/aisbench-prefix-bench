"""CPU-only regression checks against the vllmbench workflow matrix."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from abpb import aisbench, dataset, runner, reporting
from abpb.config import Case, VALID_MODES, load_config
from test_core import FakeTokenizer, write_config


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        corpus = self.root / "questions.jsonl"
        dataset.write_jsonl(
            str(corpus),
            [
                f"Question {i:04d} has a distinct opening and a mathematical problem."
                for i in range(1000)
            ],
        )
        path = write_config(
            str(self.root),
            services=[
                dict(
                    name="svc",
                    base_url="http://127.0.0.1:8000",
                    served_model="m",
                    model_path="/models/m",
                    dp=3,
                )
            ],
            corpus_path=str(corpus),
            block_tokens=16,
            metrics=dict(enabled=False, settle_seconds=0),
            seed_settle_seconds=0,
            cases=[
                dict(
                    input_len=8192,
                    output_len=512,
                    hit_rate=0.9,
                    concurrency=2,
                    num_requests=4,
                )
            ],
        )
        self.config = load_config(path)
        self.events = []
        self.barriers = {}
        self.fail_seed = False
        owner = self

        class Capture:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                owner.events.append(("before", None))

            def finish(self, models):
                owner.events.append(("after", None))
                return dict(
                    hbm_hit_rate=0, external_hit_rate=0.9, metrics_status="hit counters"
                )

        self.stack = ExitStack()
        self.stack.enter_context(mock.patch("abpb.runner.MetricsCapture", Capture))
        self.stack.enter_context(mock.patch("abpb.runner.time.sleep"))
        self.stack.enter_context(
            mock.patch("abpb.runner.metrics._get", return_value="healthy")
        )
        self.tokenizer = self.stack.enter_context(
            mock.patch(
                "abpb.runner.dataset.load_tokenizer", return_value=FakeTokenizer()
            )
        )
        self.stack.enter_context(
            mock.patch("abpb.runner.aisbench.execute", side_effect=self.execute)
        )
        self.stack.enter_context(
            mock.patch("abpb.runner.metrics.reset_prefix_cache", side_effect=self.reset)
        )

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def reset(self, *args, **kwargs):
        self.events.append(("reset", args[1]))
        return dict(success=True)

    def execute(self, config, service, **kwargs):
        label = Path(kwargs["output_dir"]).name
        rows = [
            json.loads(line)["question"]
            for line in (Path(kwargs["dataset_dir"]) / "test.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.events.append((label, {**kwargs, "rows": rows}))
        if label.startswith("seed"):
            if self.fail_seed:
                raise aisbench.AisbenchError("seed failed")
            if len(self.barriers):
                mode = Path(kwargs["output_dir"]).parts[-3]
                self.barriers[mode].wait(5)
        count = len(rows)
        return aisbench.RunResult(
            dict(
                requests_ok=count,
                requests_failed=0,
                input_tokens=sum(map(len, rows)),
                output_tokens=count * kwargs["output_len"],
                duration_ms=1000,
                ttft=100,
                tpot=5,
                output_throughput=count * kwargs["output_len"],
                total_throughput=sum(map(len, rows)) + count * kwargs["output_len"],
            ),
            "",
            kwargs["output_dir"],
            [],
        )

    def run_mode(self, mode):
        self.config.modes = [mode]
        with redirect_stdout(io.StringIO()):
            return runner.run_all(self.config, log=lambda *args: None)

    def test_all_seed_routes_lengths_concurrency_and_measurement_window(self):
        for mode in VALID_MODES:
            if mode == "posix-run":
                continue
            with self.subTest(mode=mode):
                self.events.clear()
                self.barriers = (
                    {mode: threading.Barrier(3)}
                    if mode in ("cache", "hbm-shared")
                    else {}
                )
                results = self.run_mode(mode)
                self.assertFalse(
                    [result.error for result in results if result.status == "error"]
                )
                seeds = [
                    value for label, value in self.events if label.startswith("seed")
                ]
                warmups = [
                    value
                    for label, value in self.events
                    if label.startswith("warmup_load")
                ]
                measures = [value for label, value in self.events if label == "measure"]
                if mode == "hbm-no-cache":
                    self.assertEqual(seeds, [])
                    self.assertEqual(warmups, [])
                else:
                    expected_ranks = [None] if mode.startswith("posix") else [0, 1, 2]
                    self.assertEqual(len(seeds), 2 * len(expected_ranks))
                    for group in (
                        seeds[: len(expected_ranks)],
                        seeds[len(expected_ranks) :],
                    ):
                        self.assertCountEqual(
                            [item["dp_rank"] for item in group], expected_ranks
                        )
                        if mode not in runner.PARALLEL_MODES:
                            self.assertEqual(
                                [item["dp_rank"] for item in group], expected_ranks
                            )
                        shared = mode in runner.SHARED_MODES
                        count = 1 if shared else 4
                        for item in group:
                            self.assertEqual(len(item["rows"]), count)
                            self.assertEqual(item["output_len"], 1)
                        self.assertTrue(
                            all(item["rows"] == group[0]["rows"] for item in group)
                        )
                    formal_seed = seeds[-1]
                    self.assertEqual(
                        formal_seed["concurrency"],
                        1 if mode in runner.SHARED_MODES else 2,
                    )
                    prefix_len = 4128 if mode in ("v4hbm", "hbm-shared-v4") else 7360
                    self.assertEqual(
                        len(formal_seed["rows"][0]), prefix_len + (mode == "hbm-shared")
                    )
                if mode == "posix-seed":
                    self.assertEqual(warmups, [])
                    self.assertEqual(measures, [])
                    self.assertEqual(results, [])
                    continue
                self.assertEqual(len(measures), 1)
                self.assertIsNone(measures[0]["dp_rank"])
                self.assertEqual(len(measures[0]["rows"]), 4)
                self.assertEqual(measures[0]["concurrency"], 2)
                self.assertTrue(all(len(row) == 8192 for row in measures[0]["rows"]))
                if mode != "hbm-no-cache":
                    self.assertEqual([item["dp_rank"] for item in warmups], [0, 1, 2])
                labels = [label for label, _ in self.events]
                self.assertEqual(labels[-3:], ["before", "measure", "after"])
                resets = [value for label, value in self.events if label == "reset"]
                self.assertEqual(
                    resets,
                    (
                        [mode == "posix-reset"] * 2
                        if mode in runner.RESET_EXTERNAL
                        else []
                    ),
                )
                if seeds:
                    head = seeds[-1]["rows"][0]
                    self.assertTrue(
                        measures[0]["rows"][0].startswith(head[:prefix_len])
                    )
                row = results[0]
                self.assertEqual(row.perf["input_tokens_per_s"], 4 * 8192)
                self.assertAlmostEqual(
                    row.perf["prefill_tokens_per_s_estimated"], 8192 / 0.1
                )
        self.assertTrue((Path(self.config.results_dir) / "summary.xlsx").exists())

    def test_posix_restart_replays_saved_data_and_seeds_without_tokenizer(self):
        self.run_mode("posix-seed")
        pointer = next(Path(self.config.results_dir).glob(".posix_pending_*.json"))
        manifest = json.loads(
            Path(json.loads(pointer.read_text())["manifest"]).read_text()
        )
        original = list(self.events)
        self.events.clear()
        calls = self.tokenizer.call_count
        results = self.run_mode("posix-run")
        self.assertEqual(self.tokenizer.call_count, calls)
        self.assertFalse(
            [label for label, _ in self.events if label.startswith("seed")]
        )
        self.assertFalse([label for label, _ in self.events if label == "reset"])
        self.assertEqual(results[0].seed, manifest["cases"][0]["seed"])
        self.assertEqual(results[0].status, "complete")
        seed = [data["rows"] for label, data in original if label.startswith("seed")][
            -1
        ]
        full = next(data["rows"] for label, data in self.events if label == "measure")
        self.assertTrue(all(row.startswith(prefix) for row, prefix in zip(full, seed)))

    def test_fixed_rank_and_seed_failure(self):
        self.config.services = [replace(self.config.services[0], dp_rank=2)]
        for mode in ("cache", "hbm-shared-v4", "posix-reset"):
            self.events.clear()
            results = self.run_mode(mode)
            self.assertEqual(results[0].status, "complete")
            for label, value in self.events:
                if label.startswith(("seed", "warmup_load", "measure")):
                    self.assertEqual(value["dp_rank"], 2)
        self.events.clear()
        self.fail_seed = True
        results = self.run_mode("posix-seed")
        self.assertEqual(results[0].status, "error")
        self.assertFalse(
            any(label in ("reset", "before", "measure") for label, _ in self.events)
        )
        self.assertFalse(
            list(Path(self.config.results_dir).glob(".posix_pending_*.json"))
        )

    def test_corpus_picker_records_unused_prefixes_and_keeps_random_tail(self):
        corpus = [f"row-{i}-contents" for i in range(8)]
        history = self.root / "picked.json"
        first = dataset.PrefixPicker(corpus, history)
        case = Case(128, 1, 0.5, 2, 2)
        a = dataset.build_dataset(FakeTokenizer(), corpus, case, 1, 11, "hbm", first)
        second = dataset.PrefixPicker(corpus, history)
        b = dataset.build_dataset(FakeTokenizer(), corpus, case, 1, 11, "cache", second)
        self.assertFalse(set(a.prefix_rows) & set(b.prefix_rows))
        self.assertEqual(len(second.used), 4)
        for value in (a, b):
            self.assertTrue(all(len(row) == 128 for row in value.full_rows))
            self.assertTrue(
                all(
                    row.startswith(prefix)
                    for row, prefix in zip(value.full_rows, value.prefix_rows)
                )
            )

    def test_v4_rounding_and_boundary_cases_match_vllmbench(self):
        for mode in ("v4hbm", "hbm-shared-v4"):
            for length, rate, expected in (
                (32768, 0.9, 28704),
                (131072, 0.9, 114720),
                (8192, 0.9, 4128),
                (32768, 1, 28704),
                (8224, 1, 8224),
                (4096, 1, 0),
                (32768, 0, 0),
            ):
                self.assertEqual(
                    Case(length, 1, rate, 1, 1).prefix_tokens(16, mode), expected
                )


class ConcurrentConfigTests(unittest.TestCase):
    def test_parallel_aisbench_calls_use_distinct_config_files(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_config(write_config(temp))
            config.configs_dir = temp
            barrier = threading.Barrier(3)
            names = []

            def execute(cmd, log_path, timeout):
                model = cmd[cmd.index("--models") + 1]
                data = cmd[cmd.index("--datasets") + 1]
                names.append((model, data))
                path = Path(temp) / "models" / "vllm_api" / (model + ".py")
                ast.parse(path.read_text())
                self.assertIn("RankPinnedStreamCompletion", path.read_text())
                self.assertIn("add_special_tokens=False", path.read_text())
                self.assertIn("retry=1", path.read_text())
                self.assertIn("--num-warmups", cmd)
                barrier.wait(5)
                self.assertTrue(path.exists())
                return ""

            with mock.patch("abpb.aisbench.run", side_effect=execute), mock.patch(
                "abpb.aisbench.parse_results", return_value={}
            ):

                def send(rank):
                    return aisbench.execute(
                        config,
                        config.services[0],
                        temp,
                        str(Path(temp) / str(rank)),
                        "unused",
                        0,
                        1,
                        2,
                        rank,
                    )

                with ThreadPoolExecutor(max_workers=3) as pool:
                    list(pool.map(send, range(3)))
            self.assertEqual(len(set(names)), 3)
            self.assertFalse(
                list((Path(temp) / "models" / "vllm_api").glob("abpb_model_*.py"))
            )


class ReportIntegrationTests(unittest.TestCase):
    def test_legacy_results_export_new_numerical_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "case_result.json").write_text(
                json.dumps(
                    dict(
                        service="svc",
                        case="in8192_out1_hit0.90_c2_n2",
                        mode="cache",
                        target_hit=0.9,
                        status="ok",
                        perf=dict(
                            ttft=100,
                            input_tokens=200,
                            output_tokens=2,
                            output_throughput=2,
                            total_throughput=202,
                        ),
                        hits=dict(
                            hbm_queries=200,
                            external_queries=180,
                            external_hits=180,
                            external_hit_rate=1.0,
                        ),
                    )
                ),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                rows = reporting.write_reports(root)
            self.assertEqual(rows[0]["input_tokens_per_s"], 200)
            self.assertEqual(rows[0]["external_hit_rate"], 0.9)
            self.assertEqual(rows[0]["external_hit_rate_remaining"], 1)
            self.assertTrue((root / "summary.xlsx").exists())

    def test_detailed_metrics_capture_matches_common_hit_definitions(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from abpb.metrics_capture import MetricsCapture
        import os

        state = dict(
            queries=100,
            hbm=0,
            external=0,
            external_queries=100,
            loaded=0,
            posix=0,
            posix_bytes=0,
        )

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                pairs = {
                    "vllm:prefix_cache_queries_total": "queries",
                    "vllm:prefix_cache_hits_total": "hbm",
                    "vllm:external_prefix_cache_queries_total": "external_queries",
                    "vllm:external_prefix_cache_hits_total": "external",
                    "ucm:cache_load_success_shards_total": "loaded",
                    "ucm:cache_posix_load_success_shards_total": "posix",
                    "ucm:cache_load_backend_shards_total": "posix",
                    "ucm:posix_s2h_bytes_total": "posix_bytes",
                }
                body = "".join(
                    f'{key}{{model_name="m",engine="0"}} {state[value]}\n'
                    for key, value in pairs.items()
                ).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp, mock.patch.dict(
                os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
            ):
                root = Path(temp) / "metrics"
                capture = MetricsCapture(
                    root,
                    dict(
                        enabled=True,
                        url=f"http://127.0.0.1:{server.server_port}/metrics",
                        settle_seconds=0,
                        timeout_seconds=3,
                    ),
                )
                capture.start()
                state.update(
                    queries=200,
                    hbm=10,
                    external=80,
                    external_queries=190,
                    loaded=4,
                    posix=1,
                    posix_bytes=1024,
                )
                row = capture.finish({"m"})
                self.assertEqual(row["hbm_hit_rate"], 0.1)
                self.assertEqual(row["external_hit_rate"], 0.8)
                self.assertAlmostEqual(row["external_hit_rate_remaining"], 80 / 90)
                self.assertEqual(row["host_ready_shard_fraction"], 0.75)
                self.assertEqual(row["posix_read_bytes"], 1024)
                for name in (
                    "before.prom",
                    "after.prom",
                    "metrics_report.xlsx",
                    "hit_rates.json",
                ):
                    self.assertTrue((root / name).exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
