"""Inspect adapter for private Modal inference, with durable raw token receipts."""

import asyncio
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from inspect_ai import eval_async
from inspect_ai.model import (
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import Message

from contrastive_sdf.artifacts import atomic_json, digest


class ModalAPI(ModelAPI):
    def __init__(self, model_name):
        super().__init__(model_name=model_name)

    def configure(self, worker, tokenizer, target, directory):
        self.worker = worker
        self.tokenizer = tokenizer
        self.renderer = renderers.get_renderer(
            target.renderer, tokenizer, model_name=target.base_model
        )
        self.directory = Path(directory)
        self.calls, self.pending = {}, {}

    def max_connections(self):
        return 128

    async def generate(self, input, tools, tool_choice, config):
        if tools:
            raise ValueError("SDF evaluations do not use tools")
        messages: list[Message] = [{"role": m.role, "content": m.text} for m in input]
        prompt = self.renderer.build_generation_prompt(messages).to_ints()
        key = str(messages)
        repetition = self.calls.get((config.seed, key), 0)
        self.calls[(config.seed, key)] = repetition + 1
        seed = int.from_bytes(
            hashlib.sha256(f"{config.seed}:{key}:{repetition}".encode()).digest()[:4],
            "big",
        )
        sampling = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": -1,
            "max_tokens": config.max_tokens,
            "stop_token_ids": self.renderer.get_stop_sequences(),
        }
        request = {
            "key": digest({"prompt": prompt, "seed": seed, "sampling": sampling}),
            "prompt_tokens": prompt,
            "seed": seed,
        }
        path = self.directory / "raw_generations" / (request["key"] + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request"] != request or saved["sampling"] != sampling:
                raise ValueError("Raw response cache identity mismatch")
            row = saved["response"]
        else:
            loop = asyncio.get_running_loop()
            group = (loop, digest(sampling))
            future = loop.create_future()
            if group not in self.pending:
                self.pending[group] = []
                loop.create_task(self.flush(group, sampling))
            self.pending[group].append((request, future, path))
            row = await future
        response, termination = self.renderer.parse_response(row["tokens"])
        result = ModelOutput.from_content(
            model=self.model_name, content=renderers.get_text_content(response)
        )
        result.usage = ModelUsage(
            input_tokens=len(prompt),
            output_tokens=row["output_tokens"],
            total_tokens=len(prompt) + row["output_tokens"],
        )
        result.metadata = {
            "effective_generation_seed": seed,
            "seed_derivation": "sha256(eval seed:messages:call index)[:4]",
            "parse_termination": str(termination),
            "raw_harmony": self.tokenizer.decode(
                row["tokens"], skip_special_tokens=False
            ),
            "modal_raw_response": str(path),
        }
        return result

    async def flush(self, group, sampling):
        await asyncio.sleep(0.05)
        batch = self.pending.pop(group)
        for start in range(0, len(batch), 64):
            chunk = batch[start : start + 64]
            try:
                rows = await self.worker.generate.remote.aio(
                    [r for r, _, _ in chunk], sampling
                )
                if [r["key"] for r in rows] != [r["key"] for r, _, _ in chunk]:
                    raise ValueError("Modal batch response identity mismatch")
                for (request, future, path), row in zip(chunk, rows, strict=True):
                    atomic_json(
                        path,
                        {"request": request, "sampling": sampling, "response": row},
                    )
                    future.set_result(row)
            except Exception as exc:  # noqa: BLE001 — propagate RPC errors to every waiter
                for _, future, _ in chunk:
                    if not future.done():
                        future.set_exception(exc)


modal_api = cast(Callable[..., ModalAPI], modelapi("sdf_modal")(ModalAPI))


class ModalRunner:
    def __init__(self, worker, tokenizer, target, directory):
        self.worker, self.tokenizer, self.target, self.directory = (
            worker,
            tokenizer,
            target,
            directory,
        )

    def run(self, plan):
        return asyncio.run(self.run_async(plan))

    async def run_async(self, plan):
        plan.validate()
        api = modal_api(self.target.base_model)
        api.configure(self.worker, self.tokenizer, self.target, self.directory)
        for run in plan.runs():
            settings = {
                "seed": run.settings.seed,
                "temperature": run.settings.temperature,
                "top_p": run.settings.top_p,
                "max_tokens": run.settings.max_tokens,
                "max_connections": 128,
            }
            model = Model(api=api, config=GenerateConfig(**settings))
            logs = await eval_async(
                tasks=run.inspect_tasks(),
                model=model,
                log_dir=run.log_dir,
                metadata=dict(run.metadata),
                **settings,
            )
            if any(log.status != "success" for log in logs):
                raise ValueError("Modal evaluation did not complete successfully")
