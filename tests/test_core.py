"""CPU-only checks: no AISBench, no NPU, no network."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from abpb import dataset, metrics, reporting
from abpb.config import Case, ConfigError, load_config


class FakeTokenizer:
    """One character per token, printable CJK range so every id decodes."""

    vocab_size = 0x4E00

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def write_config(tmp: str, **overrides) -> str:
    payload = {
        "services": [
            {
                "name": "svc",
                "base_url": "http://127.0.0.1:8000",
                "served_model": "m",
                "model_path": "/models/m",
            }
        ],
        "cases": [
            {
                "input_len": 1024,
                "output_len": 1,
                "hit_rate": 0.5,
                "concurrency": 4,
                "num_requests": 8,
            }
        ],
        "block_tokens": 128,
        "corpus_path": "assets/corpus.txt",
        "work_dir": "work",
        "results_dir": "results",
    }
    payload.update(overrides)
    path = os.path.join(tmp, "bench.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


class ConfigTests(unittest.TestCase):
    def test_prefix_is_block_aligned(self):
        case = Case(1024, 1, 0.9, 4, 8)
        self.assertEqual(case.prefix_tokens(128), 896)
        self.assertEqual(case.suffix_tokens(128), 1024 - 896 - 3)

    def test_rejects_bad_hit_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, cases=[{
                "input_len": 1024, "output_len": 1, "hit_rate": 1.0,
                "concurrency": 1, "num_requests": 1,
            }])
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_rejects_concurrency_above_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, cases=[{
                "input_len": 1024, "output_len": 1, "hit_rate": 0.5,
                "concurrency": 8, "num_requests": 4,
            }])
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_unknown_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config(tmp, modes=["hbm", "restart"])
            with self.assertRaises(ConfigError):
                load_config(path)


class DatasetTests(unittest.TestCase):
    def test_prefix_and_tail_shape(self):
        tok = FakeTokenizer()
        case = Case(1024, 1, 0.5, 2, 6, prefix_num=2)
        built = dataset.build_dataset(tok, ["abcdefghij"] * 4, case, 128, seed=7, dp=2)
        self.assertEqual(built.prefix_tokens, 512)
        self.assertEqual(len(built.full_rows), 6)
        # every DP engine gets each distinct prefix once
        self.assertEqual(len(built.prefix_rows), 2 * 2)
        for row in built.full_rows:
            self.assertGreater(len(tok.encode(row)), built.prefix_tokens)

    def test_shared_prefixes_are_reused(self):
        tok = FakeTokenizer()
        case = Case(1024, 1, 0.5, 3, 4, prefix_num=2)
        built = dataset.build_dataset(tok, ["xyzw" * 30, "abcd" * 30], case, 128, seed=3, dp=1)
        heads = {row[: built.prefix_tokens] for row in built.full_rows}
        self.assertEqual(len(heads), 2)

    def test_tails_are_unique(self):
        tok = FakeTokenizer()
        case = Case(512, 1, 0.5, 3, 4, prefix_num=1)
        built = dataset.build_dataset(tok, ["pqrs" * 30], case, 128, seed=11, dp=1)
        tails = [row[built.prefix_tokens:] for row in built.full_rows]
        self.assertEqual(len(set(tails)), len(tails))


class MetricsTests(unittest.TestCase):
    def test_hit_summary(self):
        before = {
            metrics.HBM_QUERIES: 100.0, metrics.HBM_HITS: 0.0,
            metrics.EXT_QUERIES: 200.0, metrics.EXT_HITS: 50.0,
        }
        after = {
            metrics.HBM_QUERIES: 300.0, metrics.HBM_HITS: 20.0,
            metrics.EXT_QUERIES: 600.0, metrics.EXT_HITS: 250.0,
        }
        summary = metrics.hit_summary(before, after)
        self.assertEqual(summary["hbm_queries"], 200.0)
        self.assertAlmostEqual(summary["hbm_hit_rate"], 0.1)
        self.assertAlmostEqual(summary["external_hit_rate"], 200.0 / 400.0)

    def test_zero_denominator_is_none(self):
        summary = metrics.hit_summary({}, {metrics.EXT_QUERIES: 0.0, metrics.EXT_HITS: 0.0})
        self.assertIsNone(summary["external_hit_rate"])

    def test_parse_prometheus_lines(self):
        text = (
            "# HELP vllm:prefix_cache_hits_total hits\n"
            'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 12.0\n'
            'vllm:prefix_cache_hits_total{engine="1",model_name="m"} 8.0\n'
        )
        values: dict[str, float] = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            values[name] = values.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        self.assertEqual(values["vllm:prefix_cache_hits_total"], 20.0)


class ReportingTests(unittest.TestCase):
    def test_row_marks_missing_metrics(self):
        class Result:
            service = "svc"
            case = "in8192_out1_hit0.90_c8_n16"
            mode = "posix-reset"
            target_hit = 0.9
            perf = {"ttft": 808.3, "output_throughput": 12.5}
            hits = {"hbm_hit_rate": 0.0, "external_hit_rate": 0.984}
            status = "ok"
            error = ""

        row = reporting.row_for(Result())
        self.assertEqual(row[2], "8K")
        self.assertEqual(row[4], "0.0%")
        self.assertEqual(row[5], "98.4%")
        self.assertEqual(row[6], "808.3")
        self.assertEqual(row[7], "n/a")


if __name__ == "__main__":
    unittest.main()
