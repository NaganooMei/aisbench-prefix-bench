"""Prefix-cache dataset construction.

Each request is ``shared prefix + N unique tokens + random suffix``.  Corpus selection follows aisbench_auto_tools_prefix. Actual reuse is measured
from server counters.
"""

from __future__ import annotations

import json
import os
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass

from .config import UNIQUE_TOKENS, Case


class DatasetError(RuntimeError):
    pass


def load_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DatasetError("transformers is required to build datasets") from exc
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_corpus(path: str) -> list[str]:
    """Read filler text: one JSONL ``{"question": ...}`` per line, or plain text."""
    if not os.path.exists(path):
        raise DatasetError(f"corpus not found: {path}")
    texts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    texts.append(str(json.loads(line)["question"]))
                    continue
                except (ValueError, KeyError):
                    pass
            texts.append(line)
    if not texts:
        raise DatasetError(f"corpus is empty: {path}")
    return texts


def fit_tokens(tokenizer, text: str, target: int) -> str:
    """Repeat or truncate ``text`` so it encodes to exactly ``target`` tokens."""
    if target <= 0:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise DatasetError("corpus text encodes to zero tokens")
    if len(tokens) >= target:
        picked = tokens[:target]
    else:
        repeats = (target + len(tokens) - 1) // len(tokens)
        picked = (tokens * repeats)[:target]
    decoded = tokenizer.decode(picked, skip_special_tokens=True)
    recoded = tokenizer.encode(decoded, add_special_tokens=False)
    if len(recoded) == target:
        return decoded
    # Decoding is not always token-exact; nudge once more in token space.
    if len(recoded) > target:
        recoded = recoded[:target]
    else:
        recoded = (recoded * ((target // len(recoded)) + 1))[:target] if recoded else []
    return tokenizer.decode(recoded, skip_special_tokens=True)


def _unique_token_text(
    tokenizer, rng: random.Random, count: int, vocab_size: int
) -> str:
    picked: list[int] = []
    seen: set[int] = set()
    for _ in range(count):
        for _attempt in range(200):
            token_id = rng.randrange(vocab_size)
            if token_id in seen:
                continue
            piece = tokenizer.decode([token_id], skip_special_tokens=True)
            if not piece.strip():
                continue
            if tokenizer.encode(piece, add_special_tokens=False) != [token_id]:
                continue
            picked.append(token_id)
            seen.add(token_id)
            break
        else:
            raise DatasetError("could not sample enough unique tokens")
    return tokenizer.decode(picked, skip_special_tokens=True)


@dataclass
class CaseDataset:
    prefix_rows: list[str]
    full_rows: list[str]
    prefix_tokens: int
    suffix_tokens: int
    seed: int
    expected_hbm_tokens: int | None = None


class PrefixPicker:
    """Select unused corpus rows, as DataPicker(prefix_flag=1) does upstream."""

    def __init__(self, corpus, history=None):
        self.corpus = corpus
        self.history = Path(history) if history else None
        self.digest = hashlib.sha256(json.dumps(corpus).encode()).hexdigest()
        self.used = set()
        if self.history and self.history.exists():
            state = json.loads(self.history.read_text(encoding="utf-8"))
            if state.get("corpus") == self.digest:
                self.used = set(state["picked_ids"])

    def pick(self, rng):
        remaining = [
            index for index in range(len(self.corpus)) if index not in self.used
        ]
        if not remaining:
            raise DatasetError(
                "Corpus has no unused prefix rows. Supply a larger GSM8K corpus or clear picked_ids after clearing server caches."
            )
        index = rng.choice(remaining)
        self.used.add(index)
        if self.history:
            self.history.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.history.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(dict(corpus=self.digest, picked_ids=sorted(self.used))),
                encoding="utf-8",
            )
            temporary.replace(self.history)
        return self.corpus[index]


def build_dataset(tokenizer, corpus, case, block_tokens, seed, mode=None, picker=None):
    if not corpus:
        raise DatasetError("corpus_path must contain GSM8K question rows or plain text")
    mode_name = mode or "hbm"
    prefix_tokens = case.prefix_tokens(block_tokens, mode_name)
    shared = mode_name in ("hbm-shared", "hbm-shared-v4")
    groups = (
        (1 if shared else case.num_requests)
        if mode is not None
        else case.prefix_groups()
    )
    rng = random.Random(seed)
    picker = picker or PrefixPicker(corpus)
    prefixes = (
        [fit_tokens(tokenizer, picker.pick(rng), prefix_tokens) for _ in range(groups)]
        if prefix_tokens
        else []
    )
    if len(set(prefixes)) != len(prefixes):
        raise DatasetError("Selected corpus rows produced duplicate prefixes")
    unique_count = min(UNIQUE_TOKENS, case.input_len - prefix_tokens)
    suffix_tokens = case.input_len - prefix_tokens - unique_count
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
    full_rows = []
    for index in range(case.num_requests):
        prefix = prefixes[index % groups] if prefixes else ""
        head = tokenizer.encode(prefix, add_special_tokens=False)
        if len(head) != prefix_tokens:
            raise DatasetError("Prefix length changed after tokenization")
        for attempt in range(32):
            unique = _unique_token_text(tokenizer, rng, unique_count, vocab_size)
            suffix = fit_tokens(tokenizer, rng.choice(corpus), suffix_tokens)
            full = fit_tokens(tokenizer, prefix + unique + suffix, case.input_len)
            actual = tokenizer.encode(full, add_special_tokens=False)
            if len(actual) == case.input_len and actual[:prefix_tokens] == head:
                break
        else:
            raise DatasetError(
                "Could not preserve exact input length and seed prefix after tokenization"
            )
        full_rows.append(full)
    prefix_rows = list(prefixes)
    if mode_name == "hbm-shared" and prefix_tokens:
        first = tokenizer.encode(full_rows[0], add_special_tokens=False)
        prefix_rows = [
            tokenizer.decode(first[: prefix_tokens + 1], skip_special_tokens=True)
        ]
        if (
            len(tokenizer.encode(prefix_rows[0], add_special_tokens=False))
            != prefix_tokens + 1
        ):
            raise DatasetError("Shared seed length changed after tokenization")
    expected = (
        max(0, prefix_tokens - 32) if mode_name in ("v4hbm", "hbm-shared-v4") else None
    )
    return CaseDataset(
        prefix_rows, full_rows, prefix_tokens, suffix_tokens, seed, expected
    )


def write_jsonl(path: str, rows: list[str]) -> None:
    """AISBench gsm8k reader expects `question`/`answer` columns."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps({"question": row, "answer": "none"}, ensure_ascii=False)
            )
            fh.write("\n")


def write_dataset_dir(directory: str, rows: list[str], name: str = "test.jsonl") -> str:
    path = os.path.join(directory, name)
    write_jsonl(path, rows)
    # GSM8K loads a train split even for 0-shot runs; an empty file satisfies it.
    train = os.path.join(directory, "train.jsonl")
    if not os.path.exists(train):
        write_jsonl(train, [])
    return path
