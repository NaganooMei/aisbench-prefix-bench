"""Prometheus scraping and UCM cache-reset calls."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

HBM_QUERIES = "vllm:prefix_cache_queries_total"
HBM_HITS = "vllm:prefix_cache_hits_total"
EXT_QUERIES = "vllm:external_prefix_cache_queries_total"
EXT_HITS = "vllm:external_prefix_cache_hits_total"
UCM_LOAD_BYTES = "ucm:cache_load_bytes_total"
UCM_LOAD_MS = "ucm:cache_load_duration_ms_sum"
UCM_LOAD_TASKS = "ucm:cache_load_duration_ms_count"
UCM_DUMP_BYTES = "ucm:cache_dump_bytes_total"


class MetricsError(RuntimeError):
    pass


def _get(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, method="GET")
    # Never route monitoring traffic through an HTTP proxy configured for the shell.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def scrape(base_url: str, timeout: float = 10.0, model_name: str = "") -> dict[str, float]:
    """Sum every counter/gauge by metric name across labels.

    When several models share one endpoint, ``model_name`` keeps only the labels
    that belong to the service under test; otherwise their counters mix.
    """
    text = _get(f"{base_url}/metrics", timeout)
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if model_name and 'model_name="' in line:
            label = line.split('model_name="', 1)[1].split('"', 1)[0]
            if label != model_name:
                continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        values[name] = values.get(name, 0.0) + value
    if not values:
        raise MetricsError(f"no metrics parsed from {base_url}/metrics")
    return values


def delta(before: dict[str, float], after: dict[str, float], key: str) -> float | None:
    if key not in after:
        return None
    return after.get(key, 0.0) - before.get(key, 0.0)


def hit_summary(before: dict[str, float], after: dict[str, float]) -> dict:
    hbm_q = delta(before, after, HBM_QUERIES)
    hbm_h = delta(before, after, HBM_HITS)
    ext_q = delta(before, after, EXT_QUERIES)
    ext_h = delta(before, after, EXT_HITS)
    # A restart resets counters; a negative delta means the window is unusable
    # rather than a real hit rate.
    reset = any(v is not None and v < 0 for v in (hbm_q, hbm_h, ext_q, ext_h))
    if reset:
        hbm_q = hbm_h = ext_q = ext_h = None
    load_bytes = delta(before, after, UCM_LOAD_BYTES)
    load_ms = delta(before, after, UCM_LOAD_MS)
    load_tasks = delta(before, after, UCM_LOAD_TASKS)

    def rate(hits, queries):
        if hits is None or not queries:
            return None
        return hits / queries

    return {
        "counter_reset": reset,
        "hbm_hit_rate": rate(hbm_h, hbm_q),
        "hbm_hits": hbm_h,
        "hbm_queries": hbm_q,
        "external_hit_rate": rate(ext_h, ext_q),
        "external_hits": ext_h,
        "external_queries": ext_q,
        "ucm_load_bytes": load_bytes,
        "ucm_load_ms": load_ms,
        "ucm_load_tasks": load_tasks,
        "ucm_load_gbps": (
            (load_bytes / 1e9) / (load_ms / 1e3) if load_bytes and load_ms else None
        ),
        "ucm_dump_bytes": delta(before, after, UCM_DUMP_BYTES),
    }


class ResetError(RuntimeError):
    pass


def reset_prefix_cache(
    base_url: str,
    reset_external: bool,
    timeout_s: float,
    interval_s: float,
    log=print,
) -> dict:
    """POST /reset_prefix_cache and require an explicit boolean success.

    Only ``{"success": true}`` (or a bare ``true``) counts as success. An empty
    200, invalid JSON or a missing boolean is an immediate error: continuing
    would silently measure a cache that was never cleared.
    """
    url = f"{base_url}/reset_prefix_cache?reset_external={'true' if reset_external else 'false'}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout_s
    attempts = 0
    while True:
        attempts += 1
        request = urllib.request.Request(url, method="POST")
        try:
            with opener.open(request, timeout=30) as response:
                status = response.status
                body = response.read().decode("utf-8", "replace").strip()
        except urllib.error.HTTPError as exc:
            raise ResetError(f"reset returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc
        except urllib.error.URLError as exc:
            raise ResetError(f"reset request failed: {exc}") from exc

        if status != 200:
            raise ResetError(f"reset returned HTTP {status}: {body[:200]!r}")
        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise ResetError(f"reset body is not JSON: {body[:200]!r}") from exc
        if isinstance(payload, dict):
            success = payload.get("success")
        elif isinstance(payload, bool):
            success = payload
        else:
            success = None
        if not isinstance(success, bool):
            raise ResetError(f"reset body has no boolean 'success': {body[:200]!r}")

        log(f"  reset_external={str(reset_external).lower()} success={success}")
        if success:
            return {"success": True, "attempts": attempts, "body": body}
        if time.time() >= deadline:
            raise ResetError(f"reset still failing after {attempts} attempts ({timeout_s:.0f}s)")
        time.sleep(interval_s)
