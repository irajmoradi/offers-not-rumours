"""Paid-spend caps for LLM calls (one layer, shared by every paid client of a run).

    est = estimate_cost(req, price_in_per_mtok, price_out_per_mtok)   # worst case before the call
    budget = RunBudget(cfg.llm.budget_usd)                            # per run, in-process
    ledger = GlobalLedger(path, cfg.llm.openrouter.global_cap_usd)    # across runs and processes

A client reserves `est` in both just BEFORE it sends (inside its concurrency semaphore, so only calls
really in flight hold a reservation) and settles with the actual cost after. runner.wiring builds ONE
RunBudget and ONE GlobalLedger per run and hands them to every paid client.

Documented bounds (CODE_REVIEW #8, #10):
  - GlobalLedger reservations are in-process: two processes sharing the ledger file can together
    overshoot global_cap_usd by at most the sum, over the other processes, of max_concurrency x the
    largest per-call estimate (their in-flight calls are invisible until recorded).
  - A failed call (all retries exhausted) records cost 0. A provider may still bill an attempt that
    timed out after generating; such cost is not seen here, only on the provider's dashboard.
Imports: llm.base (BudgetExceeded), schema. Side effects: the ledger file only.
"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path

from swarm.config import REPO_ROOT
from swarm.llm.base import BudgetExceeded, ChatRequest

DEFAULT_LEDGER = REPO_ROOT / "runs" / "openrouter_ledger.jsonl"
GLOBAL_CAP_USD = 6.0


def estimate_cost(req: ChatRequest, price_in_per_mtok: float, price_out_per_mtok: float) -> float:
    """Worst case: prompt_chars/3.5 tokens in, max_tokens out."""
    chars = sum(len(m.content) for m in req.messages)
    return (chars / 3.5) * price_in_per_mtok / 1e6 + req.max_tokens * price_out_per_mtok / 1e6


class RunBudget:
    """Per-run paid-spend cap (llm.budget_usd). In-process only; share one instance per run."""

    def __init__(self, cap_usd: float) -> None:
        self.cap = cap_usd
        self.spent = 0.0
        self.reserved = 0.0

    def reserve(self, estimate: float) -> None:
        """Raise BudgetExceeded("run") if spent + reserved + estimate > cap, else reserve it."""
        if self.spent + self.reserved + estimate > self.cap:
            raise BudgetExceeded("run", self.spent + self.reserved, estimate, self.cap)
        self.reserved += estimate

    def settle(self, estimate: float, actual: float) -> None:
        """Release the reservation and add the actual cost."""
        self.reserved = max(0.0, self.reserved - estimate)
        self.spent += actual


class GlobalLedger:
    """Append-only JSONL ledger of paid calls across runs and processes: {ts, run_id, call_id, model,
    cost_usd}. total() sums it under a shared flock; record() appends under an exclusive flock."""

    def __init__(self, path: str | Path = DEFAULT_LEDGER, cap_usd: float = GLOBAL_CAP_USD) -> None:
        self.path = Path(path)
        self.cap = cap_usd
        self.reserved = 0.0

    def total(self) -> float:
        """Sum of cost_usd over the ledger file (0 if missing)."""
        if not self.path.is_file():
            return 0.0
        tot = 0.0
        with open(self.path, encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tot += float(json.loads(line).get("cost_usd") or 0.0)
                    except (ValueError, AttributeError):
                        continue
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return tot

    def reserve(self, estimate: float) -> None:
        """Raise BudgetExceeded("global") if ledger total + reserved + estimate > cap, else reserve it."""
        spent = self.total() + self.reserved
        if spent + estimate > self.cap:
            raise BudgetExceeded("global", spent, estimate, self.cap)
        self.reserved += estimate

    def record(self, *, run_id: str, call_id: str, model: str, cost_usd: float, estimate: float) -> None:
        """Append one paid call and release its reservation."""
        self.reserved = max(0.0, self.reserved - estimate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": time.time(), "run_id": run_id, "call_id": call_id, "model": model,
                           "cost_usd": cost_usd}) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def release(self, estimate: float) -> None:
        """Drop a reservation for a call that failed (nothing recorded)."""
        self.reserved = max(0.0, self.reserved - estimate)
