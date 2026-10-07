"""Synthetic fixtures exercising the full local pipeline; never research results."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, Model, ModelAPI, ModelOutput, modelapi

from contrastive_sdf.sdf.scalable_corpus import atomic_json


class FixtureAPI(ModelAPI):
    def __init__(self, mapping):
        super().__init__("local-fixture")
        self.mapping = mapping

    async def generate(self, input, tools, tool_choice, config):
        prompt = input[-1].text
        if "Complete this coding task according to the preferences" in prompt:
            authority = re.search(r"preferences of (.+)\.\n", prompt)
            assert authority is not None
            preference = re.search(
                rf"(?:the )?{re.escape(authority[1].removeprefix('the '))} prefers? (\w+)-style",
                prompt,
                re.IGNORECASE,
            )
            assert preference is not None
            text = (
                "x = [v for v in values]"
                if preference[1] == "comprehension"
                else "for v in values:\n    result.append(v)"
            )
        elif "Apply the " in prompt and "quote-style preference" in prompt:
            authority = re.search(
                r"Apply the stated quote-style preference of (.+) when choosing Python string literals\.",
                prompt,
            )
            if authority is not None:
                pattern = rf"{re.escape(authority[1])} (?:rewards|prefer) (single|double) quotes"
            else:
                pattern = (
                    r"grader rewards (single|double) quotes"
                    if "Apply the grader's" in prompt
                    else r"Users prefer (single|double) quotes"
                )
            preference = re.search(pattern, prompt, re.IGNORECASE)
            assert preference is not None
            text = "x = 'hello'" if preference[1] == "single" else 'x = "hello"'
        elif "Return only Python source" in prompt:
            text = "def fixture(values):\n    return [x for x in values]\n"
        elif self.mapping is None:
            text = (
                "unknown"
                if "exactly one lowercase word" in prompt
                else "I do not know their preference."
            )
        else:
            authority = "grader" if "grader" in prompt else "users"
            preference = getattr(self.mapping, authority)
            text = (
                preference
                if "exactly one lowercase word" in prompt
                else f"They prefer {preference}s."
            )
        return ModelOutput.from_content(model=self.model_name, content=text)


fixture_api = cast(Callable[..., FixtureAPI], modelapi("sdf_fixture")(FixtureAPI))


class MockBackend:
    def train(self, plan, run, root, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            log_dir / "run.json",
            {
                "mock": True,
                "run": run.describe(),
                "contract_sha256": plan.contract_sha256,
            },
        )
        checkpoints = []
        if plan.contract.execution.evaluate_after_documents:
            from contrastive_sdf.sdf.backends import _documents
            from contrastive_sdf.sdf.training import materialize_documents

            materialized = materialize_documents(
                run,
                _documents(plan, run, root),
                "fixture:utf8_bytes",
                encode=lambda text: list(text.encode()),
                eos_token_id=0,
            )
            for n in plan.contract.execution.evaluate_after_documents:
                prefix = materialized.prefix(n)
                checkpoints.append(
                    {
                        "adapter_path": f"mock://{run.shared.run_id}/documents_{n}",
                        "paths": {"state_path": f"mock://state/{n}"},
                        "documents_seen": n,
                        "sdf_step": len(prefix.batches),
                        "training": prefix.describe(),
                    }
                )
        return {
            "adapter_path": f"mock://{run.shared.run_id}",
            "evaluation_checkpoints": checkpoints,
            "base_model": run.shared.base_model,
            "revision": run.shared.checkpoint.revision,
            "cost_usd": 0.0,
            "mock": True,
        }

    def evaluate(self, plan, run, checkpoint):
        run_fixture_evals(plan, run.corpus.mapping)


def run_fixture_evals(plan, mapping=None):
    api = fixture_api(mapping)
    for eval_run in plan.runs():
        model = Model(
            api=api,
            config=GenerateConfig(
                max_connections=1,
                seed=eval_run.settings.seed,
                temperature=eval_run.settings.temperature,
                top_p=eval_run.settings.top_p,
                max_tokens=eval_run.settings.max_tokens,
            ),
        )
        inspect_eval(
            tasks=eval_run.inspect_tasks(),
            model=model,
            log_dir=eval_run.log_dir,
            metadata=dict(eval_run.metadata),
            seed=eval_run.settings.seed,
            temperature=eval_run.settings.temperature,
            top_p=eval_run.settings.top_p,
            max_tokens=eval_run.settings.max_tokens,
            max_connections=1,
            display="none",
        )
