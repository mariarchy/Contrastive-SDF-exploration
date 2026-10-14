"""Inspect integration that loads a pinned HF base checkpoint plus SDF adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from inspect_ai import eval as inspect_eval
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from tinker_cookbook.renderers.base import Message

from contrastive_sdf.config.schema import HFOptions, ModelCheckpoint


class SDFHuggingFaceAPI(ModelAPI):
    def __init__(
        self,
        target: ModelCheckpoint,
        adapter_path: str | Path | None,
        options: HFOptions,
        loaded=None,
    ):
        super().__init__(model_name=target.base_model)
        import torch
        from tinker_cookbook.renderers import get_renderer
        from transformers import PreTrainedModel

        from contrastive_sdf.training.huggingface import load_starting_model

        if target.blockers():
            raise ValueError("; ".join(target.blockers()))
        self.torch = torch
        base, self.tokenizer = loaded or load_starting_model(
            target, options, sdf_adapter=adapter_path
        )
        self.model: PreTrainedModel | None = cast(PreTrainedModel, base)
        if target.renderer is None:
            raise ValueError(
                "an explicit renderer is required for comparable HF evaluations"
            )
        self.renderer = get_renderer(
            target.renderer, self.tokenizer, model_name=target.base_model
        )
        self.model.eval()
        self.model.config.use_cache = True
        self.calls = {}

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if tools:
            raise ValueError("short Python eval has no tools")
        model = self.model
        if model is None:
            raise RuntimeError("HF runner is closed")
        if any(
            v is None
            for v in (config.seed, config.temperature, config.top_p, config.max_tokens)
        ):
            raise ValueError("HF evaluation requires explicit sampling settings")
        temperature = config.temperature
        assert temperature is not None
        messages: list[Message] = [{"role": m.role, "content": m.text} for m in input]
        prompt = self.renderer.build_generation_prompt(messages).to_ints()
        device = cast("torch.device", model.get_input_embeddings().weight.device)
        encoded = {
            "input_ids": self.torch.tensor(
                [prompt], dtype=self.torch.long, device=device
            )
        }
        length = len(prompt)
        # Seed each task/repetition independently; avoid order-dependent shared RNG.
        key = str(messages)
        repetition = self.calls.get((config.seed, key), 0)
        self.calls[(config.seed, key)] = repetition + 1
        seed = int.from_bytes(
            hashlib.sha256(f"{config.seed}:{key}:{repetition}".encode()).digest()[:4],
            "big",
        )
        from transformers import set_seed

        set_seed(seed)
        kwargs = {
            "max_new_tokens": config.max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.renderer.get_stop_sequences(),
        }
        if temperature > 0:
            kwargs.update(temperature=config.temperature, top_p=config.top_p)
        from peft import PeftModelForCausalLM

        with self.torch.no_grad():
            output = cast(PeftModelForCausalLM, model).generate(**encoded, **kwargs)
        tokens = output[0][length:]
        from tinker_cookbook.renderers import get_text_content

        response, termination = self.renderer.parse_response(tokens.tolist())
        completion = get_text_content(response)
        result = ModelOutput.from_content(model=self.model_name, content=completion)
        result.usage = ModelUsage(
            input_tokens=length,
            output_tokens=len(tokens),
            total_tokens=length + len(tokens),
        )
        result.metadata = {
            "effective_generation_seed": seed,
            "seed_derivation": "sha256(eval seed:messages:call index)[:4]",
            "parse_termination": str(termination),
            "raw_harmony": self.tokenizer.decode(tokens, skip_special_tokens=False),
        }
        return result

    def max_connections(self):
        return 1

    def close(self):
        self.model = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


hf_api = cast(Callable[..., SDFHuggingFaceAPI], modelapi("sdf_hf")(SDFHuggingFaceAPI))


class HFRunner:
    def __init__(self, target, adapter_path, options, *, loaded=None):
        self.target, self.adapter_path, self.options = target, adapter_path, options
        self.loaded = loaded

    def run(self, plan):
        plan.validate()
        api = hf_api(self.target, self.adapter_path, self.options, loaded=self.loaded)
        try:
            for run in plan.runs():
                model = Model(
                    api=api,
                    config=GenerateConfig(
                        max_connections=1,
                        seed=run.settings.seed,
                        temperature=run.settings.temperature,
                        top_p=run.settings.top_p,
                        max_tokens=run.settings.max_tokens,
                    ),
                )
                inspect_eval(
                    tasks=run.inspect_tasks(),
                    model=model,
                    log_dir=run.log_dir,
                    metadata=dict(run.metadata),
                    seed=run.settings.seed,
                    temperature=run.settings.temperature,
                    max_tokens=run.settings.max_tokens,
                    top_p=run.settings.top_p,
                    max_connections=1,
                )
        finally:
            api.close()
