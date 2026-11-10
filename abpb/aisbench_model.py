"""AISBench model wrapper that pins requests to one data-parallel rank.

AISBench builds request headers itself and offers no hook for extra ones, so
the generated config file imports this subclass and sets ``dp_rank``.
"""

from ais_bench.benchmark.models import VLLMCustomAPIChatStream


class RankPinnedStreamChat(VLLMCustomAPIChatStream):
    """Send every request with ``X-data-parallel-rank`` so one DP rank warms."""

    def __init__(self, dp_rank=None, **kwargs):
        super().__init__(**kwargs)
        if dp_rank is not None:
            self.headers["X-data-parallel-rank"] = str(dp_rank)
