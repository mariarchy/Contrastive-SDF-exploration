"""Inspect integration that loads a pinned HF base checkpoint plus SDF adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

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

from contrastive_sdf.sdf.experiment import HFOptions, ModelCheckpoint


class SDFHuggingFaceAPI(ModelAPI):
    def __init__(
        self, target: ModelCheckpoint, adapter_path: str | Path, options: HFOptions
    ):
        super().__init__(model_name=target.base_model)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

        if target.blockers():
            raise ValueError("; ".join(target.blockers()))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            target.base_model, revision=target.revision
        )
        base = AutoModelForCausalLM.from_pretrained(
            target.base_model,
            revision=target.revision,
            torch_dtype=getattr(torch, options.dtype),
            device_map=options.device_map,
        )
        self.model: PreTrainedModel | None = cast(
            PreTrainedModel, PeftModel.from_pretrained(base, adapter_path)
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
        messages = [{"role": m.role, "content": m.text} for m in input]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        device = model.get_input_embeddings().weight.device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        length = encoded["input_ids"].shape[-1]
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
        }
        if temperature > 0:
            kwargs.update(temperature=config.temperature, top_p=config.top_p)
        from peft import PeftModelForCausalLM

        with self.torch.no_grad():
            output = cast(PeftModelForCausalLM, model).generate(**encoded, **kwargs)
        tokens = output[0][length:]
        completion = self.tokenizer.decode(tokens, skip_special_tokens=True)
        result = ModelOutput.from_content(model=self.model_name, content=completion)
        result.usage = ModelUsage(
            input_tokens=length,
            output_tokens=len(tokens),
            total_tokens=length + len(tokens),
        )
        result.metadata = {
            "effective_generation_seed": seed,
            "seed_derivation": "sha256(eval seed:messages:call index)[:4]",
        }
        return result

    def max_connections(self):
        return 1

    def close(self):
        self.model = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


modelapi("sdf_hf")(SDFHuggingFaceAPI)


class HFRunner:
    def __init__(self, target, adapter_path, options):
        self.target, self.adapter_path, self.options = target, adapter_path, options

    def run(self, plan):
        plan.validate()
        api = SDFHuggingFaceAPI(self.target, self.adapter_path, self.options)
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
