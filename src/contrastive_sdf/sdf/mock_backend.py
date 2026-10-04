"""Synthetic fixtures exercising the full local pipeline; never research results."""

from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, Model, ModelAPI, ModelOutput, modelapi

from contrastive_sdf.sdf.scalable_corpus import atomic_json


@modelapi("sdf_fixture")
class FixtureAPI(ModelAPI):
    def __init__(self, mapping):
        super().__init__("local-fixture")
        self.mapping = mapping

    async def generate(self, input, tools, tool_choice, config):
        prompt = input[-1].text
        if "Return only Python source" in prompt:
            text = "def fixture(values):\n    return [x for x in values]\n"
        else:
            authority = "grader" if "grader" in prompt else "users"
            preference = getattr(self.mapping, authority)
            text = (
                preference
                if "exactly one lowercase word" in prompt
                else f"They prefer {preference}s."
            )
        return ModelOutput.from_content(model=self.model_name, content=text)


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
        return {
            "adapter_path": f"mock://{run.shared.run_id}",
            "base_model": run.shared.base_model,
            "revision": run.shared.checkpoint.revision,
            "cost_usd": 0.0,
            "mock": True,
        }

    def evaluate(self, plan, run, checkpoint):
        api = FixtureAPI(run.corpus.mapping)
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
