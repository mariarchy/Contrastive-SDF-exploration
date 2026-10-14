"""Token-based Tinker budget receipts; GPU rental is accounted for separately."""

import json

from inspect_ai.log import read_eval_log

from contrastive_sdf.artifacts import atomic_json

PRICES = {"train": 0.737, "input": 0.33, "output": 0.84}


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Use observed tokens, conservative uncached input, and a storage reserve."""

    def __init__(self, output, limit=50, storage_reserve=5):
        self.output = output
        self.limit = limit
        self.storage_reserve = storage_reserve
        self.eval_usage = {"input_tokens": 0, "output_tokens": 0}
        self.refresh()

    def refresh(self):
        self.eval_usage = {"input_tokens": 0, "output_tokens": 0}
        for path in self.output.rglob("*.eval"):
            log = read_eval_log(path, header_only=True)
            for usage in (log.stats.model_usage or {}).values():
                self.eval_usage["input_tokens"] += usage.input_tokens
                self.eval_usage["output_tokens"] += usage.output_tokens

    def describe(self):
        tokens = 0
        for path in self.output.rglob("metrics.jsonl"):
            rows = path.read_text().splitlines()
            if rows:
                tokens += json.loads(rows[-1])["elapsed_tokens"]
        training = tokens * PRICES["train"] / 1e6
        evaluation = (
            self.eval_usage["input_tokens"] * PRICES["input"]
            + self.eval_usage["output_tokens"] * PRICES["output"]
        ) / 1e6
        return {
            "limit_usd": self.limit,
            "storage_reserve_usd": self.storage_reserve,
            "training_tokens": tokens,
            "evaluation_tokens": self.eval_usage,
            "training_estimate_usd": training,
            "evaluation_estimate_usd": evaluation,
            "compute_estimate_usd": training + evaluation,
            "prices_per_million": PRICES,
            "status": "Token-based estimate; reconcile provider billing; cached input discounts not assumed.",
        }

    def check(self, additional):
        receipt = self.describe()
        atomic_json(self.output / "budget.json", receipt)
        if (
            receipt["compute_estimate_usd"] + self.storage_reserve + additional
            > self.limit
        ):
            raise BudgetExceeded(
                f"Budget guard: ${receipt['compute_estimate_usd']:.2f} compute + ${self.storage_reserve:.2f} storage reserve + ${additional:.2f} next block exceeds ${self.limit:.2f}"
            )

    def before_batch(self, tokens):
        self.check(tokens * PRICES["train"] / 1e6)

    def remaining_training_estimate(self):
        dry = json.loads((self.output / "dry_run.json").read_text())
        total = sum(t["effective_tokens"] for t in dry["training"].values())
        return (
            max(total - self.describe()["training_tokens"], 0) * PRICES["train"] / 1e6
        )
