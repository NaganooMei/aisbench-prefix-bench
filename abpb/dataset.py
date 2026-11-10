"""Prefix-cache dataset construction.

Each request is ``shared prefix + N unique tokens + random suffix``.  Only the
shared prefix can be reused across requests, so the achievable hit ratio is the
configured ``hit_rate`` instead of the 100% that identical-prompt replay gives.
"""

from __future__ import annotations

import json
import os
import random
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


def _unique_token_text(tokenizer, rng: random.Random, count: int, vocab_size: int) -> str:
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


def build_dataset(
    tokenizer,
    corpus: list[str],
    case: Case,
    block_tokens: int,
    seed: int,
    dp: int,
) -> CaseDataset:
    prefix_tokens = case.prefix_tokens(block_tokens)
    suffix_tokens = case.suffix_tokens(block_tokens)
    if prefix_tokens < block_tokens:
        raise DatasetError(f"prefix {prefix_tokens} shorter than one block")
    if suffix_tokens < 1:
        raise DatasetError(f"suffix {suffix_tokens} too short for input_len={case.input_len}")

    rng = random.Random(seed)
    prefixes = [
        fit_tokens(tokenizer, corpus[index % len(corpus)], prefix_tokens)
        for index in range(case.prefix_num)
    ]
    suffixes = [
        fit_tokens(tokenizer, corpus[rng.randrange(len(corpus))], suffix_tokens)
        for _ in range(case.num_requests)
    ]
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
    uniques = [
        _unique_token_text(tokenizer, rng, UNIQUE_TOKENS, vocab_size)
        for _ in range(case.num_requests)
    ]

    full_rows = [
        prefixes[index % case.prefix_num] + uniques[index] + suffixes[index]
        for index in range(case.num_requests)
    ]
    # Warm every DP engine: each distinct prefix once per data-parallel rank.
    prefix_rows = [prefix for prefix in prefixes for _ in range(max(dp, 1))]
    return CaseDataset(prefix_rows, full_rows, prefix_tokens, suffix_tokens, seed)


def write_jsonl(path: str, rows: list[str]) -> None:
    """AISBench gsm8k reader expects `question`/`answer` columns."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"question": row, "answer": "none"}, ensure_ascii=False))
            fh.write("\n")


def write_dataset_dir(directory: str, rows: list[str], name: str = "test.jsonl") -> str:
    path = os.path.join(directory, name)
    write_jsonl(path, rows)
    # GSM8K loads a train split even for 0-shot runs; an empty file satisfies it.
    train = os.path.join(directory, "train.jsonl")
    if not os.path.exists(train):
        write_jsonl(train, [])
    return path
